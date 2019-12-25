import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os, time
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class PolicyValueNet():
    def __init__(self, board_width, board_height,block, init_model=None, transfer_model=None,cuda=False):
        print()
        print('building network ...')
        print()

        self.planes_num = 9 # feature planes
        self.nb_block = block # resnet blocks
        if cuda == False:
            # use GPU or not ,if there are a few GPUs,it's better to assign GPU ID
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif cuda == True:
            # use GPU or not ,if there are a few GPUs,it's better to assign GPU ID
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.board_width = board_width
        self.board_height = board_height
        
        self.__lock = 0
        self.__lock2 = 0

        # Make a session
        self.session = tf.InteractiveSession()
        # 1. Input:
        self.input_states = tf.placeholder(
            tf.float32, shape=[None, self.planes_num, board_height, board_width])

        self.action_fc_train, self.evaluation_fc2_train = self.network(input_states=self.input_states,
                     reuse=False,
                     is_train=True)
        self.action_fc_test,self.evaluation_fc2_test = self.network(input_states=self.input_states,
                                                                    reuse=True,
                                                                    is_train=False)

        self.network_all_params = tf.global_variables()

        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.evaluation_fc2_train)
        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(
            tf.float32, shape=[None, board_height * board_width])
        self.policy_loss = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc_train), 1)))
        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.exp(self.action_fc_test) * self.action_fc_test, 1)))

        # self.network_params = tf.trainable_variables()
        self.network_params = tf.global_variables()
        # for transfer learning use

        # For saving and restoring
        self.saver = tf.train.Saver()

        self.restore_params = []
        for params in self.network_params:
            # print(params,'**'*100)
            if ('conv2d' in params.name) or ('resnet' in params.name) or ('bn' in params.name) or ('flatten_layer' in params.name):
                self.restore_params.append(params)
        self.saver_restore = tf.train.Saver(self.restore_params)

        init = tf.global_variables_initializer()
        self.session.run(init)

        if init_model is not None:
            self.restore_model(init_model)
            print('model loaded!')
        elif transfer_model is not None:
            self.saver_restore.restore(self.session,transfer_model)
            print('transfer model loaded !')
        else:
            print('can not find saved model, learn from scratch !')
        # self.print_params()

        # opponent net for evaluating
        self.action_fc_train_oppo, self.evaluation_fc2_train_oppo = self.network(input_states=self.input_states,
                     reuse=False,
                     is_train=True,label='_oppo')
        self.action_fc_test_oppo,self.evaluation_fc2_test_oppo = self.network(input_states=self.input_states,
                                                                    reuse=True,
                                                                    is_train=False,label='_oppo')

        self.network_oppo_all_params = tf.global_variables()[len(tf.global_variables())-len(self.network_all_params):]

    def save_numpy(self,params):
        '''
        save the model in numpy form
        '''
        print('saving model as numpy form ...')
        param = []
        for each in params:
            param.append(np.array(each.eval()))
        param = np.array(param)
        np.save('tmp/model.npy',param)

    def load_numpy(self,params,path='tmp/model.npy'):
        '''
        load model from numpy
        '''
        print('loading model from numpy form ...')
        mat = np.load(path)
        for ind, each in enumerate(params):
            self.session.run(params[ind].assign(mat[ind]))
        print('load model from numpy!')

    def print_params(self,params):
        # only for debug
        return self.session.run(params)

    def policy_value(self, state_batch,actin_fc,evaluation_fc):
        '''
        input: a batch of states,actin_fc,evaluation_fc
        output: a batch of action probabilities and state values
        '''
        log_act_probs, value = self.session.run(
            [actin_fc, evaluation_fc],
            feed_dict={self.input_states: state_batch}
        )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board,actin_fc,evaluation_fc):
        '''
        input: board,actin_fc,evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        '''
        # the accurate policy value fn,
        # i prefer to use one that has some randomness even when test,
        # so that each game can play some different moves, all are ok here
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, self.planes_num, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state,actin_fc,evaluation_fc)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def policy_value_fn_random(self,board,actin_fc,evaluation_fc):
        '''
        input: board,actin_fc,evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        '''
        # like paper said,
        # The leaf node sL is added to a queue for neural network
        # evaluation, (di(p), v) = fθ(di(sL)),
        # where di is a dihedral reflection or rotation
        # selected uniformly at random from i in [1..8]
        print("rank:{}, fc: policy_value_fn_random".format(rank))
        legal_positions = board.availables
        t_rank = -1 # process in critical section

        print('rank:{}, bef fn while'.format(rank))
        while 1:
            if self.__lock==0:
                while True:
                    rewrite = 0
                    try:
                        #print('rank:{}, writing rank:{} !!!'.format(rank,rank))
                        fp = open("pv_fn.txt", "a")
                        fp.write('{}\n'.format(rank))
                        fp.close()
                    except:
                        #print('rank:{}, under reading'.format(rank))
                        #print('rank:{}, writing rank:{} !!!'.format(rank,rank))
                        time.sleep(1)
                        fp = open("pv_fn.txt", "a")
                        fp.write('{}\n'.format(rank))
                        fp.close()
                    try:
                        fp = open("pv_fn.txt", "r")
                        f_rank = fp.readline().replace('\n','')
                        #print('rank:{}, reading f_rank:{}!!!'.format(rank,f_rank))
                        if f_rank=='':
                            #print('rank:{}, type(f_rank): empty'.format(rank))
                            rewrite = 1
                        elif f_rank.isdigit():
                            #print('rank:{}, type(f_rank): int'.format(rank))
                            t_rank = int(f_rank)
                        else:
                            print('!!!!! rank:{}, type(f_rank):{}, f_rank:{}!!!!!'.format(rank,type(f_rank),f_rank))
                        fp.close()
                    except:
                        time.sleep(1)
                        #print('rank:{}, under writing'.format(rank))
                        fp = open("pv_fn.txt", "r")
                        f_rank = fp.readline().replace('\n','')
                        #print('rank:{}, reading f_rank:{}!!!'.format(rank,f_rank))
                        if f_rank=='':
                            #print('rank:{}, type(f_rank): empty'.format(rank))
                            rewrite = 1
                        elif f_rank.isdigit():
                            #print('rank:{}, type(f_rank): int'.format(rank))
                            t_rank = int(f_rank)
                        else:
                            print('!!!!! rank:{}, type(f_rank):{}, f_rank:{}!!!!!'.format(rank,type(f_rank),f_rank))
                        fp.close()
                    if rewrite == 0:
                        break

                if rank==t_rank:
                    self.__lock = 1
                    if comm.Get_size()>1:
                        for i in range(0,t_rank):
                            #print('rank:{} is now send ins: pv_fn1'.format(rank))
                            comm.send('pv_fn1',dest=i)
                        for i in range(t_rank+1,comm.Get_size()):
                            #print('rank:{} is now send ins: pv_fn1'.format(rank))
                            comm.send('pv_fn1',dest=i)
                        for i in range(0,t_rank):
                            #print('rank:{} is now send lock'.format(rank))
                            comm.send(self.__lock,dest=i)
                        for i in range(t_rank+1,comm.Get_size()):
                            #print('rank:{} is now send lock'.format(rank))
                            comm.send(self.__lock,dest=i)
                else:
                    while True:
                        print('rank:{} in pv_fn1 is now recv ins'.format(rank))
                        ins = comm.recv(source=t_rank)
                        print("rank:{} in pv_fn1 aft recv ins:{}".format(rank, ins))
                        if ins=='pv_fn1':
                            #print('rank:{} is now recv lock'.format(rank))
                            self.__lock = comm.recv(source=t_rank)
                            #print("rank:{} aft recv lock:{}".format(rank, self.__lock))
                            break
                        else:
                            print('rank:{} recv wrong'.format(rank))

            if rank == t_rank:
                current_state = np.ascontiguousarray(board.current_state().reshape(
                    -1, self.planes_num, self.board_width, self.board_height))

            if self.__lock==1:
                if rank == t_rank:
                    try:
                        print('rank:{}, file clear!!!'.format(rank))
                        self.__lock = 0
                        fp = open("pv_fn.txt", "w")
                        fp.close()
                    except:
                        time.sleep(1)
                        print('rank:{}, under reading'.format(rank))
                        print('rank:{}, file clear!!!'.format(rank))
                        self.__lock = 0
                        fp = open("pv_fn.txt", "w")
                        fp.close()
                if comm.Get_size()>1:
                    if rank==t_rank:
                        for i in range(0,t_rank):
                            #print('rank:{} is now send ins: pv_fn2'.format(rank))
                            comm.send('pv_fn2',dest=i)
                        for i in range(t_rank+1,comm.Get_size()):
                            #print('rank:{} is now send ins: pv_fn2'.format(rank))
                            comm.send('pv_fn2',dest=i)
                        for i in range(0,t_rank):
                            #print('rank:{} is now send lock'.format(rank))
                            comm.send(self.__lock,dest=i)
                        for i in range(t_rank+1,comm.Get_size()):
                            #print('rank:{} is now send lock'.format(rank))
                            comm.send(self.__lock,dest=i)
                    else:
                        while True:
                            print('rank:{} in pv_fn2 is now recv ins'.format(rank))
                            ins = comm.recv(source=t_rank)
                            print("rank:{} in pv_fn2 aft recv ins:{}".format(rank, ins))
                            if ins=='pv_fn2':
                                #print('rank:{} is now recv lock'.format(rank))
                                self.__lock = comm.recv(source=t_rank)
                                #print("rank:{} aft recv lock:{}".format(rank, self.__lock))
                                break
                            else:
                                print('rank:{} recv wrong'.format(rank))

            if rank==t_rank:
                break

        print('rank:{}, aft fn while'.format(rank))

        # print('current state shape',current_state.shape)

        #add dihedral reflection or rotation
        rotate_angle = np.random.randint(1, 5)
        flip = np.random.randint(0, 2)
        equi_state = np.array([np.rot90(s, rotate_angle) for s in current_state[0]])
        if flip:
            equi_state = np.array([np.fliplr(s) for s in equi_state])
        # print(equi_state.shape)

        print('rank:{}, bef pv while'.format(rank))
        while 1:
            if self.__lock2==0:
                while True:
                    rewrite = 0
                    try:
                        #print('rank:{}, pv writing rank:{} !!!'.format(rank,rank))
                        fp = open("pv.txt", "a")
                        fp.write('{}\n'.format(rank))
                        fp.close()
                    except:
                        #print('rank:{}, pv under reading'.format(rank))
                        #print('rank:{}, pv writing rank:{} !!!'.format(rank,rank))
                        time.sleep(1)
                        fp = open("pv.txt", "a")
                        fp.write('{}\n'.format(rank))
                        fp.close()
                    try:
                        fp = open("pv.txt", "r")
                        f_rank = fp.readline().replace('\n','')
                        #print('rank:{}, pv reading f_rank:{}!!!'.format(rank,f_rank))
                        if f_rank=='':
                            #print('rank:{}, pv type(f_rank): empty'.format(rank))
                            rewrite = 1
                        elif f_rank.isdigit():
                            #print('rank:{}, pv type(f_rank): int'.format(rank))
                            t_rank = int(f_rank)
                        else:
                            print('!!!!! rank:{}, pv type(f_rank):{}, f_rank:{}!!!!!'.format(rank,type(f_rank),f_rank))
                        fp.close()
                    except:
                        time.sleep(1)
                        #print('rank:{}, pv under writing'.format(rank))
                        fp = open("pv.txt", "r")
                        f_rank = fp.readline().replace('\n','')
                       # print('rank:{}, pv reading f_rank:{}!!!'.format(rank,f_rank))
                        if f_rank=='':
                            #print('rank:{}, pv type(f_rank): empty'.format(rank))
                            rewrite = 1
                        elif f_rank.isdigit():
                            #print('rank:{}, pv type(f_rank): int'.format(rank))
                            t_rank = int(f_rank)
                        else:
                            print('!!!!! rank:{}, pv type(f_rank):{}, f_rank:{}!!!!!'.format(rank,type(f_rank),f_rank))
                        fp.close()
                    if rewrite == 0:
                        break

                if rank==t_rank:
                    self.__lock2 = 1
                    if comm.Get_size()>1:
                        for i in range(0,t_rank):
                            #print('rank:{} is now send ins: pv1'.format(rank))
                            comm.send('pv1',dest=i)
                        for i in range(t_rank+1,comm.Get_size()):
                            #print('rank:{} is now send ins: pv1'.format(rank))
                            comm.send('pv1',dest=i)
                        for i in range(0,t_rank):
                            #print('rank:{} is now send lock'.format(rank))
                            comm.send(self.__lock2,dest=i)
                        for i in range(t_rank+1,comm.Get_size()):
                            #print('rank:{} is now send lock'.format(rank))
                            comm.send(self.__lock2,dest=i)
                else:
                    while True:
                        print('rank:{} in pv1 is now recv ins'.format(rank))
                        ins = comm.recv(source=t_rank)
                        print("rank:{} in pv1 aft recv ins:{}".format(rank, ins))
                        if ins=='pv1':
                            #print('rank:{} is now recv lock'.format(rank))
                            self.__lock2 = comm.recv(source=t_rank)
                            #print("rank:{} aft recv lock:{}".format(rank, self.__lock2))
                            break
                        else:
                            print('rank:{} recv wrong'.format(rank))

            if rank == t_rank:
                # put equi_state to network
                act_probs, value = self.policy_value(np.array([equi_state]),actin_fc,evaluation_fc)

            if self.__lock2==1:
                if rank == t_rank:
                    try:
                        print('rank:{}, pv file clear!!!'.format(rank))
                        self.__lock2 = 0
                        fp = open("pv.txt", "w")
                        fp.close()
                    except:
                        time.sleep(1)
                        print('rank:{}, pv under reading'.format(rank))
                        print('rank:{}, pv file clear!!!'.format(rank))
                        self.__lock2 = 0
                        fp = open("pv.txt", "w")
                        fp.close()
                if comm.Get_size()>1:
                    if rank==t_rank:
                        for i in range(0,t_rank):
                            #print('rank:{} is now send ins: pv2'.format(rank))
                            comm.send('pv2',dest=i)
                        for i in range(t_rank+1,comm.Get_size()):
                            #print('rank:{} is now send ins: pv2'.format(rank))
                            comm.send('pv2',dest=i)
                        for i in range(0,t_rank):
                            #print('rank:{} is now send lock'.format(rank))
                            comm.send(self.__lock2,dest=i)
                        for i in range(t_rank+1,comm.Get_size()):
                            #print('rank:{} is now send lock'.format(rank))
                            comm.send(self.__lock2,dest=i)
                    else:
                        while True:
                            print('rank:{} in pv2 is now recv ins'.format(rank))
                            ins = comm.recv(source=t_rank)
                            print("rank:{} in pv2 aft recv ins:{}".format(rank, ins))
                            if ins=='pv2':
                                #print('rank:{} is now recv lock'.format(rank))
                                self.__lock2 = comm.recv(source=t_rank)
                                #print("rank:{} aft recv lock:{}".format(rank, self.__lock2))
                                break
                            else:
                                print('rank:{} recv wrong'.format(rank))

            if rank==t_rank and self.__lock2==0:
                break

        print('rank:{}, aft pv while'.format(rank))

        # get dihedral reflection or rotation back
        equi_mcts_prob = np.flipud(act_probs[0].reshape(self.board_height, self.board_width))
        if flip:
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
        equi_mcts_prob = np.rot90(equi_mcts_prob, 4 - rotate_angle)
        act_probs = np.flipud(equi_mcts_prob).flatten()

        act_probs = zip(legal_positions, act_probs[legal_positions])
        print('rank:{}, exit pv_fn'.format(rank))
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        '''
        perform a training step
        '''
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
            [self.loss, self.entropy, self.optimizer],
            feed_dict={self.input_states: state_batch,
                       self.mcts_probs: mcts_probs,
                       self.labels: winner_batch,
                       self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        '''
        save model with ckpt form
        '''
        # only save half, without the oppo net
        self.saver.save(self.session, model_path,write_meta_graph=False)
        # write_meta_graph=False

    def restore_model(self, model_path):
        '''
        restore model from ckpt
        '''
        self.saver.restore(self.session, model_path)

    def network(self,input_states,reuse,is_train,label=''):
        # Define the tensorflow neural network
        with tf.variable_scope('model'+label, reuse=reuse):
            # tl.layers.set_name_reuse(reuse)

            input_state = tf.transpose(input_states, [0, 2, 3, 1])
            # NCHW->NHWC
            inputlayer = tl.layers.InputLayer(input_state, name='input')

            # 2. Common Networks Layers
            # these layers designed by myself
            inputlayer = tl.layers.ZeroPad2d(inputlayer,2,name='zeropad2d')
            conv1 = tl.layers.Conv2d(inputlayer,
                                          n_filter=64,
                                          filter_size=(1, 1),
                                          strides=(1, 1),
                                          padding='SAME',
                                          name='conv2d_1')
            residual_layer = self.residual_block(incoming=conv1,
                                                      out_channels=64,
                                                      is_train=is_train,
                                                      nb_block=self.nb_block)
            # 3-1 Action Networks
            # these layers are the same as paper's
            action_conv = tl.layers.Conv2d(residual_layer,
                                                n_filter=2,
                                                filter_size=(1,1),
                                                strides=(1,1),name='conv2d_2')
            action_conv = tl.layers.BatchNormLayer(action_conv,
                                                        act=tf.nn.relu,
                                                        is_train=is_train,
                                                        name='bn_1')
            action_conv_flat = tl.layers.FlattenLayer(action_conv,
                                                           name='flatten_layer_1')
            # 3-2 Full connected layer,
            # the output is the log probability of moves on each slot on the board
            action_fc = tl.layers.DenseLayer(action_conv_flat,
                                                  n_units=self.board_width*self.board_height,
                                                  act=tf.nn.log_softmax,name='dense_layer_1')
            # 4 Evaluation Networks
            # these layers are the same as paper's
            evaluation_conv = tl.layers.Conv2d(residual_layer,
                                                    n_filter=1,
                                                    filter_size=(1,1),
                                                    strides=(1,1),name='conv2d_3')
            evaluation_conv = tl.layers.BatchNormLayer(evaluation_conv,
                                                            act=tf.nn.relu,
                                                            is_train=is_train,
                                                            name='bn_2')
            evaluation_conv_flat = tl.layers.FlattenLayer(evaluation_conv,
                                                               name='flatten_layer_2')
            evaluation_fc1 = tl.layers.DenseLayer(evaluation_conv_flat,
                                                       n_units=256,
                                                       act=tf.nn.relu,
                                                       name='dense_layer_2')
            evaluation_fc2 = tl.layers.DenseLayer(evaluation_fc1,
                                                       n_units=1,
                                                       act=tf.nn.tanh,
                                                       name='flatten_layer_3')

            return action_fc.outputs,evaluation_fc2.outputs

    def residual_block(self,incoming, out_channels, is_train, nb_block=1):
        '''
        a simple resnet block structure
        '''
        resnet = incoming
        for i in range(nb_block):
            identity = resnet
            # in_channels = incoming.outputs.get_shape().as_list()[-1]
            resnet = tl.layers.Conv2d(resnet, n_filter=out_channels, filter_size=(3, 3), strides=(1, 1),
                                      padding='SAME', name='resnet_conv2d_' + str(i) + '_1')
            resnet = tl.layers.BatchNormLayer(resnet, is_train=is_train, act=tf.nn.relu,
                                              name='resnet_bn_' + str(i) + '_1')
            resnet = tl.layers.Conv2d(resnet, n_filter=out_channels, filter_size=(3, 3), strides=(1, 1),
                                      padding='SAME', name='resnet_conv2d_' + str(i) + '_2')
            resnet = tl.layers.BatchNormLayer(resnet, is_train=is_train, name='resnet_bn_' + str(i) + '_2')

            resnet = tl.layers.ElementwiseLayer([resnet, identity], combine_fn=tf.add,
                                                name='elementwise_layer_' + str(i))
            resnet = MyActLayer(resnet, act=tf.nn.relu, name='activation_layer_' + str(i))

        return resnet

class MyActLayer(Layer):
    '''
    define an activation layer
    '''
    def __init__(
        self,
        prev_layer = None,
        act = tf.identity,
        name ='activation_layer',
    ):
        Layer.__init__(self, prev_layer=prev_layer,name=name)
        self.inputs = prev_layer.outputs

        with tf.variable_scope(name) as vs:
            self.outputs = act(self.inputs)

        self.all_layers = list(prev_layer.all_layers)
        self.all_params = list(prev_layer.all_params)
        self.all_drop = dict(prev_layer.all_drop)
        self.all_layers.extend( [self.outputs])
