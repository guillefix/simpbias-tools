import torch

import torch.nn as nn

(input_dim,hidden_neurons,hidden_layers) = [int(xx) for xx in sys.argv[1:-3]]

class Net(nn.Module):
        def __init__(self, hidden_neurons, hidden_layers):
                super(Net, self).__init__()
                self.hidden_neurons, self.idden_layers = hidden_neurons, hidden_layers
                self.input_dim = input_dim
                self.layers = []
                self.layers.append(nn.Linear(self.input_dim, hidden_neurons))
                for i in range(hidden_layers):
                        self.layers.append(nn.Linear(hidden_neurons, hidden_neurons))
                self.layers.append(nn.Linear(hidden_neurons, 1))
                for i,layer in enumerate(self.layers):
                        self.add_module(str(i),layer)

        def forward(self, x):
                for layer in self.layers[:-1]:
                        x = F.relu(layer(x))
                #x = F.sigmoid(self.layers[-1](x))
                x = self.layers[-1](x)
                return x
        def get_param_vec(self):
                param_vec = []
                for f in self.parameters():
                        param_vec += f.data.view(1,-1).tolist()[0]
                return np.array(param_vec)

        # def predict(self, x):
        #       for layer in self.layers[:-1]:
        #               x = F.relu(layer(x))
        #       x = F.sign(self.layers[-1](x))
        #       return x



#asdaasfsdafsdf
