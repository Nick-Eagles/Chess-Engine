import file_IO

import _pickle as pickle
import csv
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

#   A network, training/validation data, and methods for loading and saving
#   this information correctly, bundled together
class Session:
    def __init__(self, tBuffer, vBuffer, net=None):
        self.net = net
        self.tBuffer = tBuffer
        self.vBuffer = vBuffer
        self.loadedNet = None

        #   The temporary file should be made empty when starting a Session
        with open('visualization/costs.csv', 'w') as f:
            pass

    def Save(self, dirname, data_prefix='default'):
        ########################################################################
        #   Save the network and associated attributes
        ########################################################################
        self.net.save(dirname + '/model')

        net_attributes = {
            'certainty': self.net.certainty,
            'certaintyRate': self.net.certaintyRate
        }
        with open(dirname + '/net_attributes.pkl', 'wb') as f:
            pickle.dump(net_attributes, f)

        ########################################################################
        #   Save losses to a CSV associated with the Session
        ########################################################################
        temp_losses = []
        first_new_loss = 0

        #   Copy over losses from the currently loaded network (this is None
        #   for a new network), if different than the network to save
        if self.loadedNet is not None and self.loadedNet != dirname:
            print('Reading in losses from "' + self.loadedNet + '"...')
            with open(self.loadedNet + '/costs.csv', 'r') as f:
                reader = csv.reader(f,
                                    delimiter=',',
                                    quotechar=',',
                                    quoting=csv.QUOTE_MINIMAL)

                for line in reader:
                    temp_losses.append(line)

            first_new_loss = len(temp_losses)

        #   Append temporary loss CSV to running loss CSV associated with this
        #   particular network
        print('Reading in temporary losses...')
        with open('visualization/costs.csv', 'r') as f:
            reader = csv.reader(f,
                                delimiter=',',
                                quotechar=',',
                                quoting=csv.QUOTE_MINIMAL)

            for line in reader:
                temp_losses.append(line)

        #   Overwrite temporary losses with an empty file
        print('Overwriting temporary losses with an empty file...')
        with open('visualization/costs.csv', 'w') as f:
            pass

        #   Write temporary loss to data to network-associated CSV, appending if
        #   applicable
        path = dirname +'/costs.csv'
        if (self.loadedNet is not None and os.path.isfile(path)):
            print('We will append losses to the existing ones for this net...')
            file_mode = 'a'
            temp_losses.pop(first_new_loss) # discard column names
        else:
            print('Writing a new (or overwriting) loss file for this net...')
            file_mode = 'w'
            
        with open(path, file_mode) as f:
            writer = csv.writer(f)
            writer.writerows(temp_losses)

        ########################################################################
        #   Write data buffers to CSV files
        ########################################################################

        #   Write each sub-buffer to separate file
        for i in range(4):
            tFileName = 'data/' + data_prefix + '/tBuffer' + str(i) + '.csv'
            vFileName = 'data/' + data_prefix + '/vBuffer' + str(i) + '.csv'
            file_IO.writeGames(self.tBuffer[i], tFileName, True, False)
            file_IO.writeGames(self.vBuffer[i], vFileName, True, False)

        #   The 'currently loaded net' is the one just saved
        self.loadedNet = dirname

    def Load(self, dirname, lazy=False, data_prefix='default'):
        self.net = keras.models.load_model(dirname + '/model')

        with open(dirname + '/net_attributes.pkl', 'rb') as f:
            net_attributes = pickle.load(f)

        self.net.certainty = net_attributes['certainty']
        self.net.certaintyRate = net_attributes['certaintyRate']

        if lazy:
            self.tBuffer = [[],[],[],[]]
            self.vBuffer = [[],[],[],[]]
        else:
            self.tBuffer, self.vBuffer = file_IO.loadBuffers(data_prefix)
