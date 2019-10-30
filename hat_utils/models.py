import keras.backend as K
import keras.losses as KL
import keras.optimizers as KO
from keras.layers import Input, Dense, Lambda, Activation
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

class hat_model:

    def __init__(self, model_name, save_loc, n_disc_outputs=1):
        self.disc_inputs = 0
        self.discriminant = 0
        self.adversarial = 0
        self.adversarial_2 = 0
        self.n_disc_outputs=n_disc_outputs
        self.comb_model = 0
        self.model_name = model_name
        self.save_loc = save_loc

    def create_discriminant(self, ninputs, layers=[20], dropout=0, batchnorm=0, lout=1, act="relu"):
        if len(layers) < 1:
            print('Need at least one layer! Layers passed:', layers)
            return 0

        self.disc_inputs = Input(shape=(ninputs, ))
        disc_h = Dense(layers[0])(self.disc_inputs)
        disc_h = Activation(act)(disc_h)

        for ill,ll in enumerate(layers):
            if ill == 0:
                continue
            if dropout > 0:
                disc_h = Dense(ll)(disc_h)
                disc_h = Activation(act)(disc_h)
                disc_h = Dropout(dropout_frac)(disc_h)
            elif batchnorm > 0:
                disc_h = Dense(ll)(disc_h)
                disc_h = BatchNormalization()(disc_h)
                disc_h = Activation(act)(disc_h)
            else:
                disc_h = Dense(ll)(disc_h)
                disc_h = Activation(act)(disc_h)
            
        output_activation = 'sigmoid'
        if self.n_disc_outputs > 1:
            output_activation = 'softmax'
        disc_h = Dense(self.n_disc_outputs, activation=output_activation)(disc_h)

        if lout > 0:
            disc_h = Lambda(lambda x: K.log( x/(1.0001-x)) )(disc_h)

        self.discriminant = Model(self.disc_inputs, disc_h)

        self.discriminant.summary()

        self.discriminant.compile('adam', loss=['binary_crossentropy'])

        # print("Compiling model...")
        # opt_disc = SGD(momentum=0) 
        # opt_disc = KO.Adamax()
        # self.discriminant.compile(loss=self.make_loss_discriminant(c=1.0), 
        #             optimizer=opt_disc)
        # self.discriminant.compile('adam', loss=self.make_loss_discriminant(c=1.0) )
        
        print("...Model compiled!")
    
    def create_adversarial(self, layers=[5], dropout=0, batchnorm=0, act='relu', Lambda=1.0):

        if self.discriminant == 0:
            print("Need to define discriminant network first")
            return 0
        
        if len(layers) < 1:
            print('Need at least one layer! Layers passed:', layers)
            return 0
        
        adv_h =  self.discriminant(self.disc_inputs)

        for ill,ll in enumerate(layers):
            if ill == 0:
                adv_h = Dense(ll)(adv_h)
            else:
                adv_h = Dense(ll)(adv_h)
            if dropout > 0:
                adv_h = Activation(act)(adv_h)
                adv_h = Dropout(dropout_frac)(adv_h)
            elif batchnorm > 0:
                adv_h = BatchNormalization()(adv_h)
                adv_h = Activation(act)(adv_h)
            else:
                adv_h = Activation(act)(adv_h)

        adv_h = Dense(1, activation="linear")(adv_h)

        self.adversarial = Model(self.disc_inputs, adv_h)

        print("Compiling model...")
        # opt_adv = Adam()
        # self.adversarial.compile(loss=self.make_loss_adversarial(c=Lambda), optimizer=opt_adv)
        self.adversarial.compile('adam', loss=['mean_squared_error'], loss_weights=[Lambda] )
        
        opt_adv_2 = Adam()
        self.adversarial_2 = Model( self.disc_inputs, self.adversarial(self.disc_inputs) )
        # self.adversarial_2.compile(loss=make_loss_adversarial(c=Lambda), optimizer=opt_adv_2)
        self.adversarial_2.compile('adam', loss=['mean_squared_error'], loss_weights=[Lambda] )

    def make_loss_discriminant(self, c):
        def loss_D(y_true, y_pred):
            if self.n_disc_outputs == 1:
                return c * KL.binary_crossentropy(y_pred, y_true)
                # return c * KL.binary_crossentropy(1.001/(K.exp(-y_pred) + 1), y_true)
                # return c * KL.binary_crossentropy(y_pred, y_true)
            else:
                return c * KL.sparse_categorical_crossentropy(y_pred, y_true)
        return loss_D

    def make_loss_adversarial(c):
        def loss_R(z_true, z_pred):
            return c * KL.mean_squared_error(z_pred, z_true)
        return loss_R
    
    def make_combined_model(self, Lambda=0):
        if self.discriminant == 0 or self.adversarial == 0:
            print("Need to create discriminant first, then adversarial, then combined")
            return 0
        
        self.comb_model = Model( self.disc_inputs, 
                                    [ 
                                        self.discriminant(self.disc_inputs), 
                                        self.adversarial(self.disc_inputs) 
                                    ] 
                                )
        
        # opt_comp = Adam()

        # self.comb_model.compile( loss=[self.make_loss_discriminant(c=1.0), 
        #                               self.make_loss_adversarial(c=-Lambda)], # if c=0, then no adversarial
        #                         optimizer=opt_comp )

        self.comb_model.compile( 'adam', loss=['binary_crossentropy', 
                                                'mean_squared_error'], # if c=0, then no adversarial
                                loss_weights=[1, -Lambda] )
    
    def pretrain_discriminant(self, xtrain, ytrain, wtrain=None, eps=20, batch_size=1024, patience=10):
        history = self.discriminant.fit(
            xtrain, ytrain,
            sample_weight = wtrain,
            validation_split = 0.2,
            batch_size=batch_size,
            callbacks = [
                    EarlyStopping(verbose=True, patience=patience, monitor="val_loss"),
                    ModelCheckpoint(self.save_loc + '/' + self.model_name + '_discriminant_pretraining_progress.h5', monitor="val_loss", verbose=True, save_best_only=True)
                    ],
            epochs=eps
        )

        print ('Loading best weights...')
        self.discriminant.load_weights(self.save_loc + '/' + self.model_name + '_discriminant_pretraining_progress.h5')

        print ('Saving model weights to {}...'.format(self.save_loc + '/' + self.model_name + '_discriminant_pretraining.h5'))
        self.discriminant.save_weights(self.save_loc + '/' + self.model_name + '_discriminant_pretraining.h5', overwrite=True)
        
        return history

    def pretrain_adversarial(self, xtrain, ztrain, wtrain=None, eps=20, batch_size=1024):
        self.discriminant.trainable = False
        self.adversarial.trainable = True
        history = self.adversarial_2.fit(
            xtrain, ztrain,
            sample_weight = wtrain,
            epochs=eps, batch_size=batch_size)

        return history

    def adversarial_training(self, xtrain, ytrain, ztrain, Lambda, wtrain=None, rounds=100, batch_size=1024):

        if self.comb_model == 0:
            print("Need to run make_combined_model first")
            return 0

        if type(wtrain) is not np.ndarray:
            adv_weight = None
        else:
            adv_weight = wtrain[ytrain==0]

        for r in rounds:
            print(r)
            #Fit discriminant
            self.discriminant.trainable = True
            self.adversarial.trainable = False
            self.comb_model.compile( 'adam', loss=['binary_crossentropy', 
                                                'mean_squared_error'], # if c=0, then no adversarial
                                loss_weights=[1, -Lambda] )

            indices = np.random.permutation(len(xtrain))[:batch_size]
            
            if type(wtrain) is not np.ndarray:
                comb_wtrain = None
            else:
                comb_train = (wtrain[indices], wtrain[indices])
                comb_train = list(comb_train)
            met_loss = self.comb_model.train_on_batch(xtrain[indices], [ytrain[indices], ztrain[indices]],
                                            sample_weight=comb_train)
            print(met_loss)

            # Fit adversarial
            
            self.discriminant.trainable = False
            self.adversarial.trainable = True
            self.adversarial_2.compile('adam', loss=['mean_squared_error'], loss_weights=[Lambda] )
            history = self.adversarial_2.fit(
                        xtrain[ytrain==0], ztrain[ytrain==0],
                        sample_weight = adv_weight,
                        epochs=1, batch_size=batch_size)

        print ('Saving discriminant weights to {}...'.format(self.save_loc + '/' + self.model_name + '_discriminant_adversarial_lambda_{}.h5'.format(Lambda)))
        self.discriminant.save_weights(self.save_loc + '/' + self.model_name + '_discriminant_adversarial_lambda_{}.h5'.format(Lambda))
        self.adversarial.save_weights(self.save_loc + '/' + self.model_name + '_adversarial_adversarial_lambda_{}.h5'.format(Lambda))