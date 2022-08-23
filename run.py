import json
import logging
import os
import sys
import time

import torch.optim
import math
import itertools

import models.ecr_model
import optimizers.regularizers as regularizers
from optimizers.ecr_optimizer import *
from config import parser
from utils.name2object import name2model
from rs_hyperparameter import rs_tunes, rs_hp_range, rs_set_hp_func
from gs_hyperparameter import gs_tunes, gs_hp_range, gs_set_hp_func
from utils.train import *
from utils.eval import *
from utils.visual import *
from dataset.dataset_process import preprocess_function
from models.clustring import *

from dataset.graph_dataset import GDataset, get_examples_indices
from datasets import load_dataset
from utils.bcubed_scorer import bcubed

import wandb

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizer,
    BertModel
)


def set_logger(args):
    save_dir = get_savedir(args.dataset, args.model, args.encoder, args.decoder, args.rand_search or args.grid_search)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.logs")
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    print("Saving logs in: {}".format(save_dir))
    return save_dir


def train(args, hps=None, set_hp=None, save_dir=None, num=-1, threshold=0.99):

    # config
    start_model = datetime.datetime.now()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.update({'hidden1': args.hidden2 * 2}, allow_val_change=True)
    if args.cls:
        args.update({'feat_dim': args.feat_dim * 2}, allow_val_change=True)

    if args.rand_search or args.grid_search:
        set_hp(args, hps)

    # if not (args.rand_search or args.grid_search):
    save_dir = set_logger(args)
    with open(os.path.join(save_dir, "config.json"), 'a') as fjson:
        json.dump(dict(args), fjson)

    model_name = "model_feat-d{}_h1-d{}_h2-d{}.pt".format(args.feat_dim, args.hidden1, args.hidden2)
    logging.info(args)

    if args.double_precision:
        torch.set_default_dtype(torch.float64)
        print("double precision")
    else:
        torch.set_default_dtype(torch.float32)

    # dataset###############
    dataset = GDataset(args)
    for split in ['Train', 'Dev', 'Test']:
        assert(dataset.adjacency[split].diagonal(offset=0, axis1=0, axis2=1).all()==0)

    args.n_nodes = dataset.n_nodes

    # for split in ['Train', 'Dev', 'Test']:
    #     print("###", dataset.event_chain_list[split])

    # Some preprocessing:
    # adj_norm = preprocess_adjacency(adj_train)
    pos_weight = {}
    norm = {}
    adj_norm = {}
    
    for split in ['Train', 'Dev', 'Test']:
        
        if not args.double_precision:
            dataset.adjacency[split] = dataset.adjacency[split].astype(np.float32)
        adj = dataset.adjacency[split]
        adj_norm[split] = preprocess_adjacency(adj)
        pos_weight[split] = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm[split] = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        if not args.double_precision:
            pos_weight[split] = pos_weight[split].astype(np.float32)

    event_true_sub_indices = {}
    event_false_sub_indices = {}
    entity_true_sub_indices = {}
    entity_false_sub_indices = {}
    recover_true_sub_indices = {}
    recover_false_sub_indices = {}

    #重构label不取对角线
    for split in ['Train', 'Dev', 'Test']:
        event_true_sub_indices[split], event_false_sub_indices[split] = get_examples_indices(dataset.event_coref_adj[split])
        # entity_idx = list(set(range(args.n_nodes[split])) - set(dataset.event_idx[split]))
        entity_true_sub_indices[split], entity_false_sub_indices[split] = get_examples_indices(dataset.entity_coref_adj[split])
        recover_true_sub_indices[split], recover_false_sub_indices[split] = get_examples_indices(dataset.adjacency[split])

    #Load Schema
    with open(args.schema_path, 'r') as f:  #3个set的schema
        schema_list = json.load(f)
        doc_schema = schema_list[0]
        event_schema = schema_list[1]
        entity_schema = schema_list[2]
    # bert################################
     #Load Datasets
    data_files = {}
    data_files["train"] = args.train_file
    data_files["dev"] = args.dev_file
    data_files["test"] = args.test_file
    datasets = load_dataset("json", data_files=data_files)

    #introduce PLM
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    plm = BertModel.from_pretrained(args.plm_name)

    column_names = datasets["train"].column_names
    train_dataset = datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.train_cache_file
    )

    dev_dataset = datasets["dev"]
    dev_dataset = dev_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.dev_cache_file
    )

    test_dataset = datasets["test"]
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.test_cache_file
    )

    datasets = {'Dev':dev_dataset, 'Test':test_dataset}
    ######################

    # create 
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        ValueError("WARNING: CUDA is not available!")
  
    torch.cuda.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device("cuda" if use_cuda else "cpu")

    # adj_train = torch.tensor(adj_train, device=args.device)
    # adj_norm = torch.tensor(adj_norm, device=args.device)
    # adj_orig = {}
    for split in ['Train', 'Dev', 'Test']:
        # adj_norm[split] = torch.tensor(adj_norm[split], device=args.device)
        # adj_orig[split] = torch.tensor(dataset.adjacency[split], device=args.device)
        pos_weight[split] = torch.tensor(pos_weight[split], device=args.device)

    model = getattr(models, name2model[args.model])(args, tokenizer, plm, schema_list, dataset.adjacency['Train'])
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    model.to(args.device)    # GUP

    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # regularizer = None
    # if args.regularizer:
    #     regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optimizer = GAEOptimizer(args, model, optim_method, norm, pos_weight, use_cuda)

    wandb.watch(model, log="all")

    # start train######################################
    counter = 0
    best_f1 = None
    best_epoch = None
    best_model_path = ''
    hidden_emb = None
    losses = {'Train': [], 'Dev': [], 'Test': []}
    b3s = {'Train': [], 'Dev': [], 'Test': []}
    nmis = {'Train': [], 'Dev': [], 'Test': []}
    stats = {}

    logging.info("\t ---------------------------Start Optimization-------------------------------")
    for epoch in range(args.max_epochs):
        t = time.time()
        model.train()
        if use_cuda:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        train_loss, mu = optimizer.epoch(train_dataset, adj_norm['Train'], dataset.adjacency['Train'])
        losses['Train'].append(train_loss)
        logging.info("Epoch {} | ".format(epoch))
        logging.info("\tTrain")
        logging.info("\t\taverage train loss: {:.4f}".format(train_loss))
        if math.isnan(train_loss):
            break

        # valid training set
        # hidden_emb = mu.data.detach().cpu().numpy()
        mu = mu.data.detach()
        mu = torch.sigmoid(torch.mm(mu, mu.T)).cpu()
        hidden_emb = mu

        model.eval()

        #AUC, AP###############
        metrics1 = test_model(hidden_emb, dataset.event_idx['Train'], event_true_sub_indices['Train'], event_false_sub_indices['Train'])
        logging.info("\t\tevent coref:" + format_metrics(metrics1, 'Train'))

        entity_idx = list(set(range(args.n_nodes['Train'])) - set(dataset.event_idx['Train']))
        metrics2 = test_model(hidden_emb, entity_idx, entity_true_sub_indices['Train'], entity_false_sub_indices['Train'])
        logging.info("\t\tentity coref:" + format_metrics(metrics2, 'Train'))

        metrics3 = test_model(hidden_emb, list(range(args.n_nodes['Train'])), recover_true_sub_indices['Train'], recover_false_sub_indices['Train'])
        logging.info("\t\treconstruct adj:" + format_metrics(metrics3, 'Train'))

        split = 'Train'
        wandb.log({ 
                    split+"_event_coref_auc": metrics1[0],
                    split+"_event_coref_ap": metrics1[1],
                    split+"_entity_coref_auc": metrics2[0],
                    split+"_entity_coref_ap": metrics2[1],
                    split+"_reconstruct_auc": metrics3[0],
                    split+"_reconstruct_ap": metrics3[1]
        }, commit=False)

        # B3###################
        # val#####################################
        # 无监督
        if (epoch+1) % args.valid_freq == 0:
            
            for threshold in [0.95, 0.9, 0.85]:
                # eval_model_leiden of eval_model_louvain
                #event
                logging.info("\t\tevent coref:")
                pred_list, n_comm, n_edges = eval_model_leiden(save_dir, 'Train', hidden_emb, dataset.event_idx['Train'], threshold, num)
                logging.info("\t\tthreshold={}, n_edges={}".format(threshold, n_edges))
                logging.info("\t\t\tleiden: n_community = {}".format(n_comm))
                
                eval_metrics = bcubed(dataset.event_chain_list['Train'], pred_list)
                nmi_metric = cal_nmi(dataset.event_chain_list['Train'], pred_list)
                logging.info("\t\t\tb3 metrics:" + format_b3_metrics(eval_metrics))
                logging.info("\t\t\tnmi={:.5f}".format(nmi_metric))

                comm_dect = 'leiden'
                wandb.log({
                        split+"_event_coref_"+str(threshold)+comm_dect+"_num_community": n_comm,
                        split+"_event_coref_"+str(threshold)+comm_dect+"_num_edges": n_edges,
                        split+"_event_coref_"+str(threshold)+comm_dect+"_b3_r": eval_metrics[0],
                        split+"_event_coref_"+str(threshold)+comm_dect+"_b3_p": eval_metrics[1],
                        split+"_event_coref_"+str(threshold)+comm_dect+"_b3_f": eval_metrics[2],
                        split+"_event_coref_"+str(threshold)+comm_dect+"_nmi": nmi_metric,
                }, commit=False)

                # add_new_item(stats, 'b3_r_'+str(threshold), eval_metrics[0], 'Train')
                # add_new_item(stats, 'b3_p_'+str(threshold), eval_metrics[1], 'Train')
                # add_new_item(stats, 'b3_f_'+str(threshold), eval_metrics[2], 'Train')
                # add_new_item(stats,'nmi_'+str(threshold), nmi_metric, 'Train')

                # pred_list2, n_comm2, n_edges2 = eval_model_leiden(save_dir, split, hidden_emb, dataset.event_idx[split], threshold, num)
                # logging.info("\t\tleiden: n_community = {}".format(n_comm2))
                # eval_metrics2 = bcubed(dataset.event_chain_list[split], pred_list2)
                # logging.info("\t\t\tb3 metrics:" + format_b3_metrics(eval_metrics2))

                logging.info("\t\tentity coref:")
                pred_list, n_comm, n_edges = eval_model_leiden(save_dir, 'Train', hidden_emb, dataset.entity_idx['Train'], threshold, num)
                logging.info("\t\tthreshold={}, n_edges={}".format(threshold, n_edges))
                logging.info("\t\t\tleiden: n_community = {}".format(n_comm))
                
                eval_metrics = bcubed(dataset.entity_chain_list['Train'], pred_list)
                nmi_metric = cal_nmi(dataset.entity_chain_list['Train'], pred_list)
                logging.info("\t\t\tb3 metrics:" + format_b3_metrics(eval_metrics))
                logging.info("\t\t\tnmi={:.5f}".format(nmi_metric))
                # add_new_item(stats, 'ent_b3_r_'+str(threshold), eval_metrics[0], 'Train')
                # add_new_item(stats, 'ent_b3_p_'+str(threshold), eval_metrics[1], 'Train')
                # add_new_item(stats, 'ent_b3_f_'+str(threshold), eval_metrics[2], 'Train')
                # add_new_item(stats,'ent_nmi_'+str(threshold), nmi_metric, 'Train')

                wandb.log({
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_num_community": n_comm,
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_num_edges": n_edges,
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_b3_r": eval_metrics[0],
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_b3_p": eval_metrics[1],
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_b3_f": eval_metrics[2],
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_nmi": nmi_metric,
                }, commit=False)

            for split in ['Test']:
                test_loss, test_mu = optimizer.eval(datasets[split], adj_norm[split], dataset.adjacency[split], split)  # norm adj
                losses[split].append(test_loss)
                logging.info("\t{}".format(split))
                logging.info("\t\taverage {} loss: {:.4f}".format(split, test_loss))
                wandb.log({split+'_loss': test_loss}, commit=False)

                # hidden_emb = test_mu.data.detach().cpu().numpy()
                test_mu = test_mu.detach()
                test_mu = torch.sigmoid(torch.mm(test_mu, test_mu.T)).cpu()
                hidden_emb = test_mu

                # auc, ap: test event mention pair###########
                test_metrics1 = test_model(hidden_emb, dataset.event_idx[split], event_true_sub_indices[split], event_false_sub_indices[split])
                logging.info("\t\tevent coref:" + format_metrics(test_metrics1, split))

                entity_idx = list(set(range(args.n_nodes[split])) - set(dataset.event_idx[split]))
                test_metrics2 = test_model(hidden_emb, entity_idx, entity_true_sub_indices[split], entity_false_sub_indices[split])
                logging.info("\t\tentity coref:" + format_metrics(test_metrics2, split))

                test_metrics3 = test_model(hidden_emb, list(range(args.n_nodes[split])), recover_true_sub_indices[split], recover_false_sub_indices[split])
                logging.info("\t\treconstruct adj:" + format_metrics(test_metrics3, split))

                wandb.log({
                    split+"_event_coref_auc": test_metrics1[0],
                    split+"_event_coref_ap": test_metrics1[1],
                    split+"_entity_coref_auc": test_metrics2[0],
                    split+"_entity_coref_ap": test_metrics2[1],
                    split+"_reconstruct_auc": test_metrics3[0],
                    split+"_reconstruct_ap": test_metrics3[1]
                }, commit=False)

                #b3###########################
                # logging.info("B3 Evaluation in {}:".format(split))
                for threshold in [0.95, 0.9, 0.85]:
                    # eval_model_leiden of eval_model_louvain
                    logging.info("\t\tevent coref:")
                    pred_list, n_comm, n_edges = eval_model_leiden(save_dir, split, hidden_emb, dataset.event_idx[split], threshold, num)
                    logging.info("\t\t{}, n_edges={}".format(threshold, n_edges))
                    logging.info("\t\tleiden: n_community = {}".format(n_comm))
                    
                    eval_metrics = bcubed(dataset.event_chain_list[split], pred_list)
                    nmi_metric = cal_nmi(dataset.event_chain_list[split], pred_list)
                    logging.info("\t\t\tb3 metrics:" + format_b3_metrics(eval_metrics))
                    logging.info("\t\t\tnmi={:.5f}".format(nmi_metric))
                    # add_new_item(stats, 'b3_r_'+str(threshold), eval_metrics[0], split)
                    # add_new_item(stats, 'b3_p_'+str(threshold), eval_metrics[1], split)
                    # add_new_item(stats, 'b3_f_'+str(threshold), eval_metrics[2], split)
                    # add_new_item(stats,'nmi_'+str(threshold), nmi_metric, split)

                    comm_dect = 'leiden'
                    wandb.log({
                        split+"_event_coref_"+str(threshold)+comm_dect+"_num_community": n_comm,
                        split+"_event_coref_"+str(threshold)+comm_dect+"_num_edges": n_edges,
                        split+"_event_coref_"+str(threshold)+comm_dect+"_b3_r": eval_metrics[0],
                        split+"_event_coref_"+str(threshold)+comm_dect+"_b3_p": eval_metrics[1],
                        split+"_event_coref_"+str(threshold)+comm_dect+"_b3_f": eval_metrics[2],
                        split+"_event_coref_"+str(threshold)+comm_dect+"_nmi": nmi_metric,
                    },commit=False)

                    # pred_list2, n_comm2, n_edges = eval_model_louvain(save_dir, split, hidden_emb, dataset.event_idx[split], threshold, num)
                    # logging.info("\t\tlouvain: n_community = {}".format(n_comm2))
                    # eval_metrics2 = bcubed(dataset.event_chain_list[split], pred_list2)
                    # logging.info("\t\t\tb3 metrics:" + format_b3_metrics(eval_metrics2))

                    logging.info("\t\tentity coref:")
                    pred_list, n_comm, n_edges = eval_model_leiden(save_dir, split, hidden_emb, dataset.entity_idx[split], threshold, num)
                    logging.info("\t\tthreshold={}, n_edges={}".format(threshold, n_edges))
                    logging.info("\t\t\tleiden: n_community = {}".format(n_comm))
                    
                    eval_metrics = bcubed(dataset.entity_chain_list[split], pred_list)
                    nmi_metric = cal_nmi(dataset.entity_chain_list[split], pred_list)
                    logging.info("\t\t\tb3 metrics:" + format_b3_metrics(eval_metrics))
                    logging.info("\t\t\tnmi={:.5f}".format(nmi_metric))

                    wandb.log({
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_num_community": n_comm,
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_num_edges": n_edges,
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_b3_r": eval_metrics[0],
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_b3_p": eval_metrics[1],
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_b3_f": eval_metrics[2],
                        split+"_entity_coref_"+str(threshold)+comm_dect+"_nmi": nmi_metric,
                    }, commit=False)

                    # add_new_item(stats, 'ent_b3_r_'+str(threshold), eval_metrics[0], split)
                    # add_new_item(stats, 'ent_b3_p_'+str(threshold), eval_metrics[1], split)
                    # add_new_item(stats, 'ent_b3_f_'+str(threshold), eval_metrics[2], split)
                    # add_new_item(stats,'ent_nmi_'+str(threshold), nmi_metric, split)

        wandb.log({'Train_loss': train_loss},commit=True)
        logging.info("\t\ttime={:.5f}".format(time.time() - t))
        # # 有监督
        # model.eval()
        # if (epoch + 1) % args.valid_freq == 0:
        #     # valid loss
        #     # valid metircs
        #     metrics = test_model()   # F1
        #     logging.info("\t Epoch {} | average valid loss: {:.4f}".format(epoch, valid_loss))
        #     logging.info(format_conll('Valid_F1'+valid_f1))  
        #     if not best_f1 or valid_f1 > best_f1:
        #         best_f1 = valid_f1
        #         counter = 0
        #         best_epoch = epoch
        #         logging.info("\t Saving model at epoch {} in {}".format(epoch, save_dir))
        #         best_model_path = os.path.join(save_dir, '{}_{}'.format(epoch, model_name))
        #         torch.save(model.cpu().state_dict(), best_model_path)
        #         if use_cuda:
        #             model.cuda()

        #     else:
        #         counter += 1
        #         if counter == args.patience:
        #             logging.info("\t Early stopping")
        #             break
        #         elif counter == args.patience // 2:
        #             pass
        # ###################

        #save_freq###########
        # if (epoch+1) % args.save_freq == 0 or (epoch + 1)==args.max_epochs:

        #     model_path = os.path.join(save_dir, str(epoch+1)+model_name)
            
        #     save_check_point(model, model_path)
        #     # torch.save(model.cpu().state_dict(), model_path)
        #     # model.to(args.device)

    logging.info("\t ---------------------------Optimization finished---------------------------")

    # # test#########################
    # if not best_f1:
    #     best_model_path = os.path.join(save_dir, model_name)
    #     torch.save(model.cpu().state_dict(), best_model_path)
    # else:
    #     logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
    #     model.load_state_dict(torch.load(best_model_path))  # load best model
    # if use_cuda:
    #     model.cuda()
    # model.eval()  # no BatchNormalization Dropout

    # # Test metrics

    # 测评
    # logging.info("Evaluation Test Set:")
    # test_f1 = None
    # conll_f1 = run_conll_scorer(args.output_dir)
    # logging.info(conll_f1)

    model_path = os.path.join(save_dir, model_name)  #最后一个epoch
    # save_check_point(model, model_path)
    torch.save(model.state_dict(), model_path)
    wandb.save('model.h5')
    # torch.save(model.cpu().state_dict(), model_path)

    # plot(save_dir, 'converg', num, losses['Train'], losses['Dev'], losses['Test'])
    # plot1(save_dir, 'converg', num, losses['Train'])

    # str_metrics = ['b3_r','b3_p','b3_f','nmi']
    # str_threshs = ['0.95','0.9', '0.85']
    # descrips = itertools.product(str_metrics, str_threshs)
    # for des in descrips:
    #     s = des[0] + '_' + des[1]
    #     plot_splits(save_dir, s, num, stats[s])
    #     plot_splits(save_dir, 'ent_'+s, num, stats['ent_'+s])

    # # save statistics to file
    # res = []
    # for vdict in [losses, b3s, nmis]:
    #     if len(vdict['Test'])> 0 :
    #         res.append(vdict)
    #         plot(save_dir, 'converg', num, vdict['Train'], vdict['Dev'], vdict['Test'])

    # # save statistics to file
    # stat_dir = os.path.join(save_dir, str(num)+'statistics'+'.json')
    # with open(stat_dir, 'w') as f:
    #     json.dump(stats, f)
    #     f.close()

    end_model = datetime.datetime.now()
    logging.info('this model runtime: %s' % str(end_model - start_model))
    logging.info("\t ---------------------------done---------------------------")
    return None


def rand_search(args):

    best_f1 = 0
    best_hps = []
    best_f1s = []

    save_dir = set_logger(args)
    logging.info("** Random Search **")

    args.tune = rs_tunes
    logging.info(rs_hp_range)
    hyperparams = args.tune.split(',')

    if args.tune == '' or len(hyperparams) < 1:
        logging.info("No hyperparameter specified.")
        sys.exit(0)
    grid = rs_hp_range[hyperparams[0]]
    for hp in hyperparams[1:]:
        grid = zip(grid, rs_hp_range[hp])

    grid = list(grid)
    logging.info('* {} hyperparameter combinations to try'.format(len(grid)))

    for i, grid_entry in enumerate(list(grid)):
        if not (type(grid_entry) is list):
            grid_entry = [grid_entry]
        grid_entry = flatten(grid_entry)    # list
        hp_values = dict(zip(hyperparams, grid_entry))
        logging.info('* Hyperparameter Set {}:'.format(i))
        logging.info(hp_values)

        test_metrics = train(args, hp_values, rs_set_hp_func, save_dir, i)
        logging.info('{} done'.format(grid_entry))
    #     if test_metrics['F'] > best_f1:
    #         best_f1 = test_metrics['F']
    #         best_f1s.append(best_f1)
    #         best_hps.append(grid_entry)
    # logging.info("best hyperparameters: {}".format(best_hps))


def grid_search(args):

    best_f1 = 0
    best_hps = []
    best_f1s = []

    save_dir = set_logger(args)
    logging.info("** Grid Search **")

    args.tune = gs_tunes
    logging.info(gs_hp_range)
    hyperparams = args.tune.split(',')

    if args.tune == '' or len(hyperparams) < 1:
        logging.info("No hyperparameter specified.")
        sys.exit(0)
    grid = gs_hp_range[hyperparams[0]]
    for hp in hyperparams[1:]:
        grid = itertools.product(grid, gs_hp_range[hp])

    grid = list(grid)
    logging.info('* {} hyperparameter combinations to try'.format(len(grid)))

    for i, grid_entry in enumerate(list(grid)):
        if not (type(grid_entry) is list):
            grid_entry = [grid_entry]
        grid_entry = flatten(grid_entry)    # list
        hp_values = dict(zip(hyperparams, grid_entry))
        logging.info('* Hyperparameter Set {}:'.format(i))
        logging.info(hp_values)

        test_metrics = train(args, hp_values, gs_set_hp_func, save_dir, i)
        logging.info('{} done'.format(grid_entry))
    #     if test_metrics['F'] > best_f1:
    #         best_f1 = test_metrics['F']
    #         best_f1s.append(best_f1)
    #         best_hps.append(grid_entry)
    # logging.info("best hyperparameters: {}".format(best_hps))


if __name__ == "__main__":
    start = datetime.datetime.now()

    wandb.init(project="ecr-gsl-baseline-test", config=parser)
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
    args = wandb.config

    if parser.rand_search:
        rand_search(args)
    else:
        if parser.grid_search:
            grid_search(args)
        else:
            train(args)

    end = datetime.datetime.now()
    logging.info('total runtime: %s' % str(end - start))
    sys.exit()
