"""Functions for each command."""

import time
import logging
from collections import deque
import numpy
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import *
import operator 
import shutil

import dataprep
# from .rl import *
from util import *
from agent import *




    
def train(model, args):
    r"""Train the model.
    
    Implement step-by-step training. Use it to train models with step
    dependency.

    Args:
        model (torch.nn.Module): model to be trained
        args (namedtuple): training parameters. See :class:`~stn.run.Train`.
    """

    # set model to train mode
    model.train()

    # load training snapshot if needed
    # TODO: also save/load optimizer state
    if args.snapshot is None:
        start_epoch = 0
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
        logging.info('Loaded ' + str(epoch))
        start_epoch = int(epoch.split('.')[-1])

    # build spotlight control module if needed
    if args.spotlight_model == 'markov':
        agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
    elif args.spotlight_model == 'rnn':
        agent = RNNPolicy(model.hidden_size + model.img_size + 3, 64)
    else:
        agent = None

    model.to(device)
    if agent:
        agent.to(device)

    # load spotlight control module snapshot if needed
    if args.focus is not None:
        load_snapshot(agent, args.workspace, args.focus)
        logging.info('Loaded ' + str(args.focus))

    # load and split data
    logging.info('Loading data...')
    data, cat = dataprep.get_kdd_dataset(args.dataset, args.spotlight_model)
    if args.split_frac > 0:
        data, _ = data.split(args.split_frac)

    optim = torch.optim.Adam(model.parameters(),
                             weight_decay=args.norm)
    if agent:
        output_optim = torch.optim.Adam(model.output.parameters(),
                                        weight_decay=args.norm)
        agent_optim = torch.optim.Adam(agent.parameters(),
                                       weight_decay=args.norm)

    then = time.time()

    # training
    for epoch in range(start_epoch, args.epochs):
        total = len(data.keys)
        N = 0  # sequence count

        total_loss = 0.
        total_agent_loss = 0.
        n = 0  # item count, for averaging

        for keys, item in data.shuffle().epoch(args.batch_size,
                                               backend='torch'):
            N += len(keys)

            # Due to a bug in yata, sequences with same lens will be packed as
            # a regular tensor. For consistency below, turn tensors into padded
            # sequences.
            if type(item.y) != tuple:
                sentences = item.y
                lens = [item.y.size(1)] * item.y.size(0)
            else:
                sentences, lens = item.y

            img_input = item.file.to(device)
            hs, h_imgs, ss = model.get_initial_state(img_input)

            # loss of current batch
            loss = 0.
            agent_loss = 0.

            # forward step for each sequence
            for k in range(len(keys)):
                sentence = sentences[k:k + 1, :lens[k]]
                sentence = sentence.to(device)
                L = sentence.size(1)  # sequence length

                # special tokens
                # TODO: follow seq2seq naming convension
                null = torch.zeros(1, 1).long().to(device)
                beg = null + 1

                # in PyTorch, batch is at dim 1 for sequences
                x = torch.cat([beg, sentence], dim=1).permute(1, 0)
                y_true = torch.cat([sentence, null], dim=1).permute(1, 0)

                h = hs[:, k:k + 1, :]
                h_img = h_imgs[k:k + 1, :, :, :]
                s = None if ss is None else ss[k:k + 1, :]

                if agent:
#                    sh = None
                    sh = agent.default_h().to(device)
                    c = agent.default_c().to(device)

                for i in range(L + 1):
                    n += 1
                    if agent:
                        # spotlight transcribing network
                        h = model.get_h(x[i:i + 1, :], h)
                        c = torch.cat([h.view(1, -1),
                                       c[:, model.hidden_size:]], dim=1)
                        if len(item) == 5:
                            # item with position supervision
                            if i != L:
                                cx, cy = item.pos[k][i][1:-1].split(',')
                                cx = float(cx)
                                cy = float(cy)
                                sigma = -3.
                                s_true = torch.tensor([[cx, cy, sigma]])
                            else:
                                # EOS has no position, set it to previous s
                                s_true = s.detach()

                            s_pred, sh = agent(s, c.detach(), sh)
                            agent_loss += F.smooth_l1_loss(s_pred, s_true)
                            # set s_true as actual input
                            y_pred, h, alpha, c = \
                                model.put_h(h, h_img, s_true)
                        
                            s = s_true

                        else:
                            # TODO: add an argument to control whether agent
                            # and model is completely separated
                            s, sh = agent(s, c.detach(), sh)
                            y_pred, h, alpha, c = model.put_h(h, h_img, s)

                    else:
                        # attention-based methods
                        y_pred, h, alpha, _ = model(x[i:i + 1, :], h, h_img)

                    loss += F.cross_entropy(y_pred, y_true[i])
            total_loss += loss.item()
            if type(agent_loss) != float:
                total_agent_loss += agent_loss.item()

            optim.zero_grad()
            if agent:
                agent_optim.zero_grad()

            loss.backward()
            if type(agent_loss) != float:
                agent_loss.backward()

            optim.step()
            if agent:
                agent_optim.step()

            loss = 0.
            agent_loss = 0.

            now = time.time()
            duration = (now - then) / 60
            then = now

            # TODO: use a modified version of tqdm to show progress
            logging.info('[%d:%d/%d] (%.2f samples/min) '
                         'loss %.6f, agent_loss %.6f' %
                         (epoch, N, total, args.batch_size / duration,
                          total_loss / n,
                          total_agent_loss / n))

        if args.refine_each_epoch:
            for keys, item in data.sample(0.05).epoch(args.batch_size,
                                                      backend='torch'):
                # same as above
                # TODO: refactor duplicated code
                N += len(keys)

                if type(item.y) != tuple:
                    sentences = item.y
                    lens = [item.y.size(1)] * item.y.size(0)
                else:
                    sentences, lens = item.y

                img_input = item.file.to(device)
                hs, h_imgs, ss = model.get_initial_state(img_input)

                # forward step for each sequence
                for k in range(len(keys)):
                    sentence = sentences[k:k + 1, :lens[k]]
                    L = sentence.size(1)  # sequence length

                    # special tokens
                    # TODO: follow seq2seq naming convension
                    null = torch.zeros(1, 1).long().to(device)
                    beg = null + 1

                    # in PyTorch, batch is at dim 1 for sequences
                    x = torch.cat([beg, sentence], dim=1) \
                        .permute(1, 0).to(device)
                    y_true = torch.cat([sentence, null], dim=1) \
                        .permute(1, 0).to(device)

                    h = hs[:, k:k + 1, :]
                    h_img = h_imgs[k:k + 1, :, :, :]
                    s = None if ss is None else ss[k:k + 1, :]

                    if agent:
                        sh = agent.default_h().to(device)
                        c = agent.default_c().to(device)

                    for i in range(L + 1):
                        n += 1
                        if agent:
                            # spotlight transcribing network
                            h = model.get_h(x[i:i + 1, :], h)
                            c = torch.cat([h.view(1, -1),
                                           c[:, model.hidden_size:]], dim=1)

                            if len(item) == 5:
                                # item with position supervision
                                if i != L:
                                    cx, cy = item.pos[k][i][1:-1].split(',')
                                    cx = float(cx)
                                    cy = float(cy)
                                    sigma = -3.
                                    s_true = torch.tensor([[cx, cy, sigma]])
                                else:
                                    # EOS has no position, set it to previous s
                                    s_true = s.detach()

                                s_pred, sh = agent(s, c.detach(), sh)
                                agent_loss += F.smooth_l1_loss(s_pred, s_true)

                                # set s_true as actual input
                                y_pred, h, alpha, c = \
                                    model.put_h(h, h_img, s_true)

                                s = s_true

                            else:
                                s, sh = agent(s, c.detach(), sh)
                                y_pred, h, alpha, c = model.put_h(h, h_img, s)

                        else:
                            # attention-based methods
                            y_pred, h, alpha, _ = model(x[i:i + 1, :],
                                                        h, h_img)

                        loss += F.cross_entropy(y_pred, y_true[i])

                total_loss += loss.item()
                if type(agent_loss) != float:
                    total_agent_loss += agent_loss.item()

                # only optimize output layer and agent
                output_optim.zero_grad()
                agent_optim.zero_grad()

                loss.backward()
                if type(agent_loss) != float:
                    agent_loss.backward()

                output_optim.step()
                agent_optim.step()

                loss = 0.
                agent_loss = 0.

                now = time.time()
                duration = (now - then) / 60
                then = now

                logging.info('[%d:%d/%d] (%.2f samples/min) '
                             'loss %.6f, agent_loss %.6f' %
                             (epoch, N, total, args.batch_size / duration,
                              total_loss / n,
                              total_agent_loss / n))

        save_snapshot(model, args.workspace, 'main.' + str(epoch + 1))
        if agent:
            save_snapshot(agent, args.workspace, 'agent.' + str(epoch + 1))


def train_batched(model, args):
    r"""Train the model in batches.
    
    Implement batched training. Use it to train models without step dependency.

    Args:
        model (torch.nn.Module): model to be trained
        args (namedtuple): training parameters. See :class:`~stn.run.Batched`.
    """

    # set model to train mode
    model.train()

    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))

    data, cat = dataprep.get_kdd_dataset(args.dataset)
    if args.split_frac > 0:
        data, _ = data.split(args.split_frac)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.norm)
    start_epoch = load_last_snapshot(model, args.workspace)
    then = time.time()

    for epoch in range(start_epoch, args.epochs):
        try:
            logging.info(('epoch {}:'.format(epoch)))
            N = len(data.keys)
            n = 0
            total_loss = 0.
            time_n = 0
            for keys, item in data.shuffle().epoch(args.batch_size,
                                                   backend='torch'):
                if len(item.y) != 2:
                    sentences = item.y
                    lens = [item.y.size(1)] * item.y.size(0)
                else:
                    sentences, lens = item.y

                plens = [l + 1 for l in lens]

                sentences = sentences.to(device)
                bz = len(keys)
                nulls = torch.zeros(bz, 1).long().to(device)
                begs = nulls + 1

                x = torch.cat([begs, sentences], dim=1).permute(1, 0)
                y_true = torch.cat([sentences, nulls], dim=1).permute(1, 0)
                y_true_ = pack_padded_sequence(y_true, plens)

                y_pred_ = model.pred_on_batch(item.file.to(device), x, plens)

                loss = F.cross_entropy(y_pred_, y_true_.data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n += len(keys)
                time_n += len(keys)
                total_loss += loss.item() * len(keys)

                now = time.time()
                if now - then < 5:
                    continue
                duration = (now - then) / 60
                then = now
                logging.info('[%d:%d/%d] (%.2f samples/min) '
                             'loss %.6f' %
                             (epoch, n, N, time_n / duration,
                              total_loss / n))
                time_n = 0

            save_snapshot(model, args.workspace, epoch + 1)

        except KeyboardInterrupt as e:
            save_snapshot(model, args.workspace, 'int')
            raise e
def test(model, args):
    r"""Test the model.

    Args:
        model (torch.nn.Module): model to be tested
        args (namedtuple): training parameters. See :class:`~stn.run.Test`.

    .. todo:
        Use beam search.
    """

    # set model to test mode
    model.eval()
    torch.set_grad_enabled(False)

    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))

    logging.info('Loading data...')
    data, cat = dataprep.get_kdd_dataset(args.dataset, args.focus)
    if args.split_frac > 0:
        _, data = data.split(args.split_frac)

    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)

    if args.focus:
        if args.spotlight_model == 'markov':
            agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
        else:
            agent = RNNPolicy(model.hidden_size + model.img_size + 3, 64)
        load_snapshot(agent, args.workspace, args.focus)

    logging.info('loaded model at epoch %s', str(epoch))

    model.to(device)

    N = len(data.keys)
    n = 0
    total_acc = 0.

    for keys, item in data.shuffle().epoch(1, backend='torch'):
        n += 1

        sentence = item.y
        L = sentence.size(1)

        null = torch.zeros(1, 1).long()
        beg = null + 1

        x = torch.cat([beg, sentence], dim=1).permute(1, 0)
        y_true = torch.cat([sentence, null], dim=1).permute(1, 0)
        y_pred = torch.zeros(L * 2 + 21, 1).long().to(device)
        img = item.file.to(device)
        h, h_img, s = model.get_initial_state(img)

        same = 0
        y_ = x[0:1, :]

        if args.focus:
            sh = agent.default_h().to(device)
            c = agent.default_c().to(device)

        true_seq = []
        pred_seq = []
        for i in range(L * 2 + 20):
            if args.focus:
                h = model.get_h(y_, h)
                c = torch.cat([h.view(1, -1), c[:, model.hidden_size:]], dim=1)
                s, sh = agent(s, c, sh)
                y_, h, alpha, c = model.put_h(h, h_img, s)
            else:
                y_, h, _, _ = model(y_, h, h_img)
            y_ = y_.max(1)[1]
            y_pred[i, 0] = y_
            if i <= L:
                true_seq.append(y_true[i, 0])
                same += (y_ == y_true[i, 0]).item()
            pred_seq.append(y_.item())
            y_ = y_.unsqueeze(0)
        total_acc += lcs(true_seq, pred_seq) / len(true_seq) \
            if args.lcs else same / (L + 1)
        if n % 50 == 0 or n == len(data.keys):
            logging.info('[%d/%d] acc %.6f' %
                         (n, N, total_acc / n))

    torch.set_grad_enabled(True)
        
def beamroadtest(seq):

    
    
    model = seq["param"][0]
    agent = seq["param"][1]
    args  = seq["param"][2]
    x_    = seq["param"][3]
    h_    = seq["param"][4]
    c_    = seq["param"][5]
    s_    = seq["param"][6]
    sh_   = seq["param"][7]
    h_img_ = seq["param"][8]
    y_true = seq["param"][9]

    
    if args.focus:
        h_ = model.get_h(x_, h_)
        c_ = torch.cat([h_.view(1, -1), c_[:, model.hidden_size:]], dim=1)
        s_, sh_ = agent(s_, c_, sh_)
        y_, h_, alpha, c_ = model.put_h(h_, h_img_, s_)
    else:
        y_, h_, alpha, _ = model(x_, h_, h_img_)


            
    x = []
    p = {}
    q = {}
    y = y_.numpy()[0]
    x.append(var(torch.from_numpy(numpy.array([numpy.argsort(y)[-1]]))).unsqueeze(0))
    x.append(var(torch.from_numpy(numpy.array([numpy.argsort(y)[-2]]))).unsqueeze(0))
    p["seq"] = seq["seq"] + [numpy.array([numpy.argsort(y)[-1]])[0]]
    p["weight"] = seq["weight"] + y[numpy.argsort(y)[-1]]/(y[numpy.argsort(y)[-1]]+abs(y[numpy.argsort(y)[0]]))
    p["same"] = seq["same"] + ( torch.from_numpy(numpy.array([numpy.argsort(y)[-1]])) == y_true[len(p["seq"])-1, 0]).item()
    p["param"]  = [model, agent,args,x[0],h_,c_,s_,sh_,h_img_,y_true]

        
    q["seq"] = seq["seq"] + [numpy.array([numpy.argsort(y)[-2]])[0]]
    q["weight"] = seq["weight"] +y[numpy.argsort(y)[-2]]/(y[numpy.argsort(y)[-1]]+abs(y[numpy.argsort(y)[0]]))
    q["same"] = seq["same"] +  ( torch.from_numpy(numpy.array([numpy.argsort(y)[-2]])) == y_true[len(q["seq"])-1, 0]).item()
    q["param"]  = [model, agent,args,x[1],h_,c_,s_,sh_,h_img_,y_true]
    roads = []
    roads.append(p)
    roads.append(q)
        
    return roads
    torch.set_grad_enabled(True)
    

def testbeam(model, args):
    r"""Test the model.

    Args:
        model (torch.nn.Module): model to be tested
        args (namedtuple): training parameters. See :class:`~stn.run.Test`.

    .. todo:
        Use beam search.
    """

    # set model to test mode
    model.eval()
    torch.set_grad_enabled(False)

    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))

    logging.info('Loading data...')
    data, cat = dataprep.get_kdd_dataset(args.dataset, args.focus)
    if args.split_frac > 0:
        _, data = data.split(args.split_frac)

    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)

    if args.focus:
        if args.spotlight_model == 'markov':
            agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
        else:
            agent = RNNPolicy(model.hidden_size + model.img_size + 3, 64)
        load_snapshot(agent, args.workspace, args.focus)

    logging.info('loaded model at epoch %s', str(epoch))



    N = len(data.keys)
    n = 0
    total_acc = 0.
    cat = dataprep.get_cat(model.args.words)
    
    for keys, item in data.shuffle().epoch(1, backend='torch'):
        n += 1
        sentence = item.y
        sentence0 = item.y.numpy()[0]
        L = sentence.size(1)
        null = torch.zeros(1, 1).long()
        beg = null + 1

        x = torch.cat([beg, sentence], dim=1).permute(1, 0)
        y_true = torch.cat([sentence, null], dim=1).permute(1, 0)
        y_pred = torch.zeros(L * 2 + 21, 1).long().to(device)
        img = item.file.to(device)
        h, h_img, s = model.get_initial_state(img)

        same = 0
        y_ = x[0:1, :]

        if args.focus:
            sh = agent.default_h()
            c = agent.default_c()

        true_seq = []
        pred_seq = []
        #初始轮
        if args.focus:
            h = model.get_h(y_, h)
            c = torch.cat([h.view(1, -1), c[:, model.hidden_size:]], dim=1)
            s, sh = agent(s, c, sh)
            y_, h, alpha, c = model.put_h(h, h_img, s)
        else:
            y_, h, _, _ = model(y_, h, h_img)
        y = y_.numpy()[0]
        p = {}
        q = {}

        
        x = []
        x.append(var(torch.from_numpy(numpy.array([numpy.argsort(y)[-1]]))).unsqueeze(0))

        p["seq"] =[]
        p["seq"].append(numpy.array([numpy.argsort(y)[-1]])[0])
        
        p["weight"] = y[numpy.argsort(y)[-1]]/(y[numpy.argsort(y)[-1]]+abs(y[numpy.argsort(y)[0]]))
        p["same"] =  ( torch.from_numpy(numpy.array([numpy.argsort(y)[-1]])) == y_true[0, 0]).item()
        p["param"] = [model, agent,args,x[0],h,c,s,sh,h_img,y_true]
        
        #第一轮
        roadstmp = beamroadtest(p)
        result = []
        roads = []
        for i in range(0,2):
            if roadstmp[i]["seq"][-1] == 0 or len(roadstmp[i]["seq"]) == L+1:
                result.append(roadstmp[i])
            else:
                roads.append(roadstmp[i])
        #后续轮
        while len(roads) > 0:
            k = len(roads)
            tmp = []
            for i in range(0,k):
                if roads[i]["seq"][-1] != 0:
                    road1,road2 = beamroadtest(roads[i])
                    tmp.append(road1)
                    tmp.append(road2)
            tmp.sort(key=operator.itemgetter("weight"),reverse=True)
            roads = []
            for i in range(0,k):
                if tmp[i]["seq"][-1] == 0 or len(tmp[i]["seq"]) == L+1:
                    del tmp[i]["param"]
                    result.append(tmp[i])
                else:
                    roads.append(tmp[i])

            

            
        
        if result[0]["weight"] > result[1]["weight"]:
            pred_seq = result[0]["seq"]
            same = result[0]["same"]

        else:
            pred_seq =result[1]["seq"]
            same = result[1]["same"]
        
#        for i in range(0,len(y_true)):
#            true_seq.append(y_true[i, 0].item())
            #print("road2")
        if same / (L+1) == 1:
            print(keys[0]+".png")
            shutil.copyfile('C:\\Users\\msi\\Desktop\\Service\\data\\formula\\'+keys[0]+".png",'C:\\Users\\msi\\Desktop\\Service\\testdata_formula\\'+keys[0]+".png")
        total_acc += lcs(true_seq, pred_seq) / len(true_seq) \
            if args.lcs else same / (L+1)

        
        if n % 50 == 0 or n == len(data.keys):
            logging.info('[%d/%d] acc %.6f' %
                         (n, N, total_acc / n))
    
        
    torch.set_grad_enabled(True)


def lcs(a, b):
    """Longest common substring of two strings."""
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j],
                                            lengths[i][j + 1])
    # read the substring out from the matrix
    result = 0
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result += 1
            x -= 1
            y -= 1
    return result


def examine(model, args):
    r"""Check outputs of the model.

    Args:
        model (torch.nn.Module): model to be examined
        args (namedtuple): training parameters. See :class:`~stn.run.Examine`.
    """
    model.eval()
    torch.set_grad_enabled(False)
    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))

    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)

    if args.dataset == 'melody':
        data, chars = dataprep.get_melody()
    else:
        data, chars = dataprep.get_formula()
    _, data = data.split(0.9)

    if args.focus:
        if args.spotlight_model == 'markov':
            agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
        else:
            agent = RNNPolicy(model.hidden_size + model.img_size + 3, 32)
        load_snapshot(agent, args.workspace, args.focus)

    if use_cuda:
        model.cuda()

    n = 0

    for keys, item in data.shuffle().epoch(1, backend='torch'):
        if n == 50:
            break
        n += 1

        sentence = item.y
        L = sentence.size(1)

        null = torch.zeros(1, 1).type_as(sentence)
        beg = torch.zeros(1, 1).type_as(sentence) + 1

        x = var(torch.cat([beg, sentence], dim=1), volatile=True).permute(1, 0)
        y_true = torch.cat([sentence, null], dim=1).permute(1, 0)
        y_pred = var(torch.zeros(L + 21, 1).type(torch.LongTensor))

        h, h_img, s = model.get_initial_state(var(item.file, volatile=True))
        if args.focus:
            sh = agent.default_h()
            c = agent.default_c()

        if args.focus:
            s, sh = agent(s, c, sh)
            y_, h, alpha, c = model(x[0:1, :], h, h_img, s)
        else:
            y_, h, alpha = model(x[0:1, :], h, h_img)
        y_pred[0, 0] = y_.squeeze().max(0)[1]

        same = y_pred.data[0, 0] == y_true[0, 0]
        for i in range(L + 20):
            if args.focus:
                s, sh = agent(s, c, sh)
                y_, h, alpha, c = model(y_pred[i:i + 1, :], h, h_img, s)
            else:
                y_, h, alpha = model(y_pred[i:i + 1, :], h, h_img)
            out = y_.squeeze().max(0)[1]
            y_pred[i + 1, 0] = out
            if same and out.data[0] != y_true[i + 1, 0]:
                same = False
            if out.data[0] == 0:
                break

        if same:
            color = bcolors.OKGREEN
        else:
            color = bcolors.FAIL

        print(colored('> ', bcolors.OKGREEN, True) +
              ''.join(chars.get_original(y_true[:-1].squeeze())))
        print(colored('< ', color, True) +
              ''.join(chars.get_original(y_pred.squeeze()[:i + 1].data)))
        print()

    torch.set_grad_enabled(True)

def visualize(model, args):


    model.eval()
    torch.set_grad_enabled(False)
    seq = ""
    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
        logging.info('Loaded epoch: ' + epoch)

    agent_hs = args.agent_hs
    if args.focus:
        if args.spotlight_model == 'markov':
            agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
        else:
            agent = RNNPolicy(model.hidden_size + model.img_size + 3, agent_hs)
        load_snapshot(agent, args.workspace, args.focus)

    cat = dataprep.get_cat(model.args.words)
    img = dataprep.load_img(args.input, (args.W, args.H),
                            not args.colored).unsqueeze(0)
    img = var(img, volatile=True)
    h, h_img, s = model.get_initial_state(img)
    if args.focus:
        sh = agent.default_h()
        c = agent.default_c()
    x = var(torch.zeros(1, 1).long() + 1)
#    import matplotlib.pyplot as plt
    for _ in range(40):

        if args.focus:
            h = model.get_h(x, h)
            c = torch.cat([h.view(1, -1), c[:, model.hidden_size:]], dim=1)
            s, sh = agent(s, c, sh)
            y, h, alpha, c = model.put_h(h, h_img, s)
        else:
            y, h, alpha, _ = model(x, h, h_img)

#        if args.focus:
#            v = torch.zeros(3)
#            v[:] = s.squeeze().data
#            v[2:] = torch.sigmoid(v[2:])
#            v.clamp_(0, 1)
        y = y.data.max(1)[1]

        seq += cat.get_original(y)[0]
        seq += " "
#        fig = plt.figure()
#        ax = fig.add_subplot(111)

        a = -alpha[0]
        position = numpy.where(a.data.numpy()==numpy.min(a.data.numpy()))
        radius = numpy.where(a.data.numpy()[position[0][0]]<=numpy.min(a.data.numpy()[position[0][0]])/2)

        print(y,position[0][0],position[1][0],numpy.max(abs(radius[0] - position[1][0])),cat.get_original(y)[0])

#        cax = ax.matshow(a.data.numpy(), cmap='RdYlBu',
#                         interpolation='gaussian')
#        fig.colorbar(cax)
#        fig.show()
#        plt.show()
        sys.stdout.flush()
        time.sleep(1)
        if y[0] == 0:
            break
        x = var(y).unsqueeze(0)
    print("test:"+seq)
    
    torch.set_grad_enabled(True)
    
    
def beamroad(seq):
    
    
    
    model = seq["param"][0]
    agent = seq["param"][1]
    args  = seq["param"][2]
    x_    = seq["param"][3]
    h_    = seq["param"][4]
    c_    = seq["param"][5]
    s_    = seq["param"][6]
    sh_   = seq["param"][7]
    h_img_ = seq["param"][8]

    cat = dataprep.get_cat(model.args.words)
    if args.focus:
        h_ = model.get_h(x_, h_)
        c_ = torch.cat([h_.view(1, -1), c_[:, model.hidden_size:]], dim=1)
        s_, sh_ = agent(s_, c_, sh_)
        y_, h_, alpha, c_ = model.put_h(h_, h_img_, s_)
    else:
        y_, h_, alpha, _ = model(x_, h_, h_img_)


            
    x = []
    p = {}
    q = {}
    y = y_.numpy()[0]
    v = y_.numpy()[0]
    a = -alpha[0]

    
    
    position = numpy.where(a.data.numpy()==numpy.min(a.data.numpy()))
    position2 = [numpy.array([numpy.argsort(a.data.numpy(),axis = 0).T[position[1]][0][1]]),position[1]]

    radius = numpy.where(a.data.numpy()[position[0][0]]<=numpy.min(a.data.numpy()[position[0][0]])/2)
    radius2 = numpy.where(a.data.numpy()[position2[0][0]]<=a.data.numpy()[position2[0][0]][position2[1][0]]/2)
    y = y_.numpy()[0]
 
    x.append(var(torch.from_numpy(numpy.array([numpy.argsort(y)[-1]]))).unsqueeze(0))
    x.append(var(torch.from_numpy(numpy.array([numpy.argsort(y)[-2]]))).unsqueeze(0))
    word =cat.get_original(torch.from_numpy(numpy.array([numpy.argsort(y)[-1]])))[0]
    if word == "":
        p["seq"] += "blank"
    else:
        p["seq"] = seq["seq"] + word
        
    p["weight"] = seq["weight"] + y[numpy.argsort(y)[-1]]/(y[numpy.argsort(y)[-1]]+abs(y[numpy.argsort(y)[0]]))
    p["spot"] = seq["spot"] + [[position[0][0],position[1][0],numpy.max(abs(radius[0] - position[1][0]))]]
    p["seq"] += " "
    p["param"]  = [model, agent,args,x[0],h_,c_,s_,sh_,h_img_]        

    word =cat.get_original(torch.from_numpy(numpy.array([numpy.argsort(y)[-2]])))[0]
    if word == "":
        q["seq"] += "blank"
    else:
        q["seq"] = seq["seq"] + word
    q["weight"] = seq["weight"] +y[numpy.argsort(y)[-2]]/(y[numpy.argsort(y)[-1]]+abs(y[numpy.argsort(y)[0]]))
    q["spot"] = seq["spot"] + [[position2[0][0],position2[1][0],numpy.max(abs(radius2[0] - position2[1][0]))]]
    q["seq"] += " "
    q["param"]  = [model, agent,args,x[1],h_,c_,s_,sh_,h_img_]    
    roads = []
    roads.append(p)
    roads.append(q)
        
    return roads
'''
    
        v = y.numpy()[0]    
        y = y.data.max(1)[1]
        position = numpy.where(a.data.numpy()==numpy.min(a.data.numpy()))
        radius = numpy.where(a.data.numpy()[position[0][0]]<=numpy.min(a.data.numpy()[position[0][0]])/2)
        seq["spot"].append([position[0][0],position[1][0],numpy.max(abs(radius[0] - position[1][0]))])
        seq["seq"] += cat.get_original(y)[0]
        seq["seq"] += " "

    torch.set_grad_enabled(True)
    return seq
'''
def visualizebeam(model, args):
    model.eval()
    torch.set_grad_enabled(False)
    seq = ""
    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
        logging.info('Loaded epoch: ' + epoch)

    agent_hs = args.agent_hs
    if args.focus:
        if args.spotlight_model == 'markov':
            agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
        else:
            agent = RNNPolicy(model.hidden_size + model.img_size + 3, agent_hs)
        load_snapshot(agent, args.workspace, args.focus)

    cat = dataprep.get_cat(model.args.words)
    img = dataprep.load_img(args.input, (args.W, args.H),
                            not args.colored).unsqueeze(0)
    img = var(img, volatile=True)
    h, h_img, s = model.get_initial_state(img)
    if args.focus:
        sh = agent.default_h()
        c = agent.default_c()
    t = var(torch.zeros(1, 1).long() + 1)
        
    #初始轮
    if args.focus:
        h = model.get_h(t, h)
        c = torch.cat([h.view(1, -1), c[:, model.hidden_size:]], dim=1)
        s, sh = agent(s, c, sh)
        y_, h, alpha, c = model.put_h(h, h_img, s)
    else:
        y_, h, alpha, _ = model(t, h, h_img)

    y = y_.numpy()[0]
    p = {}
    q = {}
    x = []
    a = -alpha[0]
    position = numpy.where(a.data.numpy()==numpy.min(a.data.numpy()))
    radius = numpy.where(a.data.numpy()[position[0][0]]<=numpy.min(a.data.numpy()[position[0][0]])/2)
    x.append(var(torch.from_numpy(numpy.array([numpy.argsort(y)[-1]]))).unsqueeze(0))
    p["seq"] =cat.get_original(torch.from_numpy(numpy.array([numpy.argsort(y)[-1]])))[0]
    if p["seq"] == "":
        p["seq"] += "blank"
    p["weight"] = y[numpy.argsort(y)[-1]]/(y[numpy.argsort(y)[-1]]+abs(y[numpy.argsort(y)[0]]))
    p["spot"] = []
    p["spot"].append([position[0][0],position[1][0],numpy.max(abs(radius[0] - position[1][0]))])
    p["seq"] += " "
    p["param"] = [model, agent,args,x[0],h,c,s,sh,h_img]

    
    #第一轮
    roadstmp = beamroad(p)
    result = []
    roads = []
    for i in range(0,2):
        if roadstmp[i]["seq"].split()[-1] == "<NULL>":
            result.append(roadstmp[i])
        else:
            roads.append(roadstmp[i])
    #后续轮
    while len(roads) > 0:
        k = len(roads)
        tmp = []
        for i in range(0,k):
            road1,road2 = beamroad(roads[i])
            tmp.append(road1)
            tmp.append(road2)
        tmp.sort(key=operator.itemgetter("weight"),reverse=True)
        roads = []
        for i in range(0,k):
            if tmp[i]["seq"].split()[-1] == "<NULL>":
                del tmp[i]["param"]
                result.append(tmp[i])
            else:
                roads.append(tmp[i])

            

            
        
    if result[0]["weight"] > result[1]["weight"]:
        
        print(result[0])


    else:
        print(result[1])
    
    torch.set_grad_enabled(True)


def reinforce(model, args):
    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))

    # load pretrained model
    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
    logging.info('Loaded epoch: ' + str(epoch))
    logging.info('Loading data...')

    if args.dataset == 'melody':
        data, cat = dataprep.get_melody()
    else:
        data, cat = dataprep.get_formula()
    data, _ = data.split(0.9)

    learner = ActorCritic(model)

    if use_cuda:
        model.cuda()

    for i in range(args.times):
        logging.info('start %dth time', i)
        learner.reinforce(data.shuffle(), cat)
        save_snapshot(model, args.workspace, 'rf.' + str(i))
        # TODO: validate


# deprecated
def pretrain_attention(model, args):
    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
    logging.info('Loaded epoch: ' + str(epoch))

    if args.spotlight_model == 'markov':
        agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
    else:
        agent = RNNPolicy(model.hidden_size + model.img_size + 3, 32)

    logging.info('Loading data...')
    if args.dataset == 'melody':
        data, cat = dataprep.get_melody()
    else:
        data, cat = dataprep.get_formula()
    data, _ = data.split(args.split_frac)

    if use_cuda:
        model.cuda()
        agent.cuda()

    model.suspend_focus = True

    optim = torch.optim.Adam(model.parameters(), weight_decay=args.norm)
    agent_optim = torch.optim.Adam(agent.parameters(), weight_decay=args.norm)

    for epoch in range(args.epochs):
        N = 0
        loss = 0.
        agent_loss = 0.

        total_loss = 0.
        total_agent_loss = 0.
        n = 0
        then = time.time()

        for keys, item in data.shuffle().epoch(1, backend='torch'):
            N += 1

            sentence = item.y
            L = sentence.size(1)

            null = torch.zeros(1, 1).type_as(sentence)
            beg = torch.zeros(1, 1).type_as(sentence) + 1
            x = var(torch.cat([beg, sentence], dim=1)).permute(1, 0)
            y_true = var(torch.cat([sentence, null], dim=1).permute(1, 0))

            h, h_img, s = model.get_initial_state(var(item.file))
            sh = agent.default_h()
            c = agent.default_c()

            for i in range(L + 1):
                n += 1
                s_pred, sh = agent(s, var(c.data), sh)
                y_pred, h, alpha, c = model(x[i:i + 1, :], h, h_img)
                loss += F.cross_entropy(y_pred, y_true[i])

                s = var(c[:, -3:].data)
                agent_loss += F.smooth_l1_loss(s_pred, s)

            if N % args.batch_size == 0:
                total_loss += loss.data[0]
                total_agent_loss += agent_loss.data[0]

                optim.zero_grad()
                loss.backward()
                optim.step()

                agent_optim.zero_grad()
                agent_loss.backward()
                agent_optim.step()

                loss = 0.
                agent_loss = 0.

                now = time.time()
                duration = (now - then) / 60
                logging.info('[%d:%d] (%.2f samples/min) '
                             'loss %.6f, agent_loss %.6f' %
                             (epoch, N, args.batch_size / duration,
                              total_loss / n,
                              total_agent_loss / n))
                then = now

        save_snapshot(model, args.workspace, 'focus.' + str(epoch + 1))
        save_snapshot(agent, args.workspace, 'agent.pre.' + str(epoch + 1))
