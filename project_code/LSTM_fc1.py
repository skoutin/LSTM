import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import signal_handler
import matplotlib.pyplot as plt

run_to_gpu = 0
if run_to_gpu : device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_model= 0
save_model = 0
load_model = not (train_model)


#------------------------------------------------------------------------------------------------------------------------------------------------------------
### data preperation

# Import data
# lfp_data = signal_handler.extract_all_data(downsampling=10000, save_path='D:/Files/peirama_dipl/project_files/LSTM_fc_data.npy') 
lfp_data = np.load('D:/Files/peirama_dipl/project_files/LSTM_fc_data.npy')
plt.plot(lfp_data[0,:250]), plt.show() # σχεδιασμός ενός 20λεπτου για να έχεις εικόνα πως μοιάζει το σήμα
print(lfp_data.shape) # έχεις 56 αρχεία 20 λεπτων και πρέπει να εκπαιδεύσεις τα βάρη σε όλα τα αρχεία ανά 100 σημεία

# ο αριθμός που καθορίζει πόσα στοιχεία θα χρησιμοποιηθουν για το forecasting του επόμενου σημείου. 
# Τα δεδομένα εκπαίδεσης (δηλαδή τα παράθυρα) θα φτιαχτούν βάσει αυτού του αριθμού
fc_num = 250

# το for loop ενώνει όλα τα παράθυρα σε ένα πίνακα 3 διαστάσεων που μπορεί να γίνει input για το LSTM
# το windowing κόβει το σήμα σε παράθυρα των (fc_num+1) σημείων με overlap 96%. 
# το windowing2 κόβει το σήμα σε παράθυρα των (fc_num+1) σημείων αλλά το κόψιμο γίνεται με slidng του παραθυρου και όχι με overlaping
# Τα fc_num σημεία θα είναι input_data και το (fc_num+)ο σημείο θα είναι το target data. 
data=[]
for i in np.arange(lfp_data.shape[0]):
    # s=signal_handler.windowing(lfp_data[i,:], fc_num+1, 0.96) 
    s=signal_handler.windowing2(lfp_data[i,:], fc_num+1, 1)
    data.append(s)
data=np.stack(data)
print(data.shape)

data = torch.from_numpy(data).float() # transform to a tensor
data = F.normalize(data, dim = 0) # normalizes data, hopefully in the right axis (δηλαδή στη forecasting dimension με τιμή fc_num)
input_data = data[:,:, 0:fc_num]
target_data = data[:,:,fc_num] # με πρόλεψη επόμενου ενός σημείου
# target_data = data[:,:,1:fc_num+1] # με πρόλεψη ίδιου αριθμού σημείων μετατοποισμένων κατά ένα σημείο
if run_to_gpu : input_data = input_data.to(device); target_data = target_data.to(device) 
dataset = torch.utils.data.TensorDataset(input_data, target_data)
train_data, val_data = torch.utils.data.dataset.random_split(dataset, [0.9, 0.1]) 
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=train_data.__len__())
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=1)

#------------------------------------------------------------------------------------------------------------------------------------------------------------
## initialization of LSTM-based-neural-network parameters
input = fc_num # the input size of the lstm -> it will take 100 (more generally fc_num) points of time series
hidden_state_dim = 30 # the size of the hidden/cell state of LSTM
num_layers = 5 # the number of consecutive LSTM cells the nn.LSTM will have
output_size = 1 # the final output of the NN. Here it will take a 100 points and predict the 101 point of the time series
# output_size = fc_num # να δοκιμάσεις να προβλέψεις τον ίδιο αριθμόο σημείων μετατοπισμένο κατά ένα δες γραμμή κώδικα 56

#------------------------------------------------------------------------------------------------------------------------------------------------------------
## Architecture of the LSTM-based-neural-network
class LSTM_fc1(nn.Module): 
    """this model will be a forecasting LSTM model that takes 100 (or more) points and finds some points in the future. 
    How many are the 'some' points depndes from the output size and the target data """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_fc1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers

        self.lstm=nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        # h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, requires_grad=True)
        # c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, requires_grad=True)
        # if run_to_gpu : h_t = h_t.to(device); c_t = c_t.to(device)
        # out, (h_t, c_t) = self.lstm(x, (h_t, c_t)) -> ήταν έτσι αν αρχικοποιούσες τα h_t και c_t, αλλά μάλλον είναι καλύτερα χωρίς αυτά
        out, (h_t, c_t) = self.lstm(x) # out dims (batch_size, L, hidden_size) if batch_first=True
        out = self.linear(out)
        return out
#------------------------------------------------------------------------------------------------------------------------------------------------------------
## NN instance creation
lstm_model = LSTM_fc1(input, hidden_state_dim, num_layers, output_size)
if run_to_gpu : lstm_model = lstm_model.to(device)
criterion = nn.MSELoss()
optimizer=optim.SGD(lstm_model.parameters(), lr=0.05)

### try forward method with a (εχεις φτιάξει ένα LSTM που παίρνει ένα τενσορα 100(fc_num)) στοιχείων και επιστρέφει ένα τενσορα 1 στοιχείου)
# a=np.linspace(0,3,fc_num); a=torch.tensor(a, dtype=torch.float32); a=torch.unsqueeze(a,0); a=torch.unsqueeze(a,0);print(a.shape)
# arr = lstm_model(a) # input must be dims (batch_size, sequence_length, input_size)
# print(arr.shape); print(arr)

#------------------------------------------------------------------------------------------------------------------------------------------------------------
if train_model:
    ### train the model
    num_epochs = 50
    lstm_model.train()
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            # lstm_model.train()
            train_pred = lstm_model(x_batch)
            train_pred = torch.squeeze(train_pred) # ΑΥΤΟ ΜΠΟΡΕΙ ΝΑ ΜΗΝ ΕΙΝΑΙ ΣΩΣΤΟ -> TELIKA EINAI
            # print(y_batch - train_pred) # αν η διαφορά είναι πολύ μικρή ίσως τα δεδομένα χρειάζονται κανονικοποίηση
            loss = criterion (y_batch, train_pred) #!!! the user warning problem arises here
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch:{epoch+1}/{num_epochs} -> Loss = {loss.item()}')

    ### validation
    with torch.no_grad():
        val_losses =[]
        val_predictions=[]
        lstm_model.eval()
        for x_val, y_val in val_loader:
            test_pred = lstm_model(x_val)
            test_pred = torch.squeeze(test_pred) # ΑΥΤΟ ΜΠΟΡΕΙ ΝΑ ΜΗΝ ΕΙΝΑΙ ΣΩΣΤΟ -> TELIKA EINAI
            y_val = torch.squeeze(y_val) # ΑΥΤΟ ΜΠΟΡΕΙ ΝΑ ΜΗΝ ΕΙΝΑΙ ΣΩΣΤΟ
            val_loss = criterion (y_val, test_pred) #!!! the user warning problem arises here
            val_losses.append(val_loss.item())
        print('Validation loss is', val_losses)

    ## save the model
    if save_model : torch.save(lstm_model.state_dict(), 'D:/Files/peirama_dipl/project_files/LSTM_forecasting.pt')

# load the model if you have saved it in order no to run training agian if its time-consuming
if load_model:
    model = LSTM_fc1(input, hidden_state_dim, num_layers, output_size) 
    model.load_state_dict(torch.load('D:/Files/peirama_dipl/project_files/LSTM_forecasting.pt'))
    model.eval()
else:
    model = lstm_model

#------------------------------------------------------------------------------------------------------------------------------------------------------------

def generate_lfp(model, starting_signal,num_fc_points):
    """""1) model is the LSTM forecasting model
        2) starting_signal must be an lfp signal in tensor form of lstm input length
        3) num_fc_point is the number of the points that willbe generated/forecasted
    """""
    model = model.to('cpu')
    starting_signal = starting_signal.to('cpu')
    model.eval()
    # generated_signal= list(starting_signal.cpu().numpy()) # αν θέλουμε το παραγώμενο σήμα να περιέχει το input
    generated_signal=[] # αν θέλουμε το παραγώμενο σήμα να περιέχει μονο το generated χωρίς το input
    for i in range(num_fc_points):
        starting_signal_input=torch.unsqueeze(starting_signal,0)
        starting_signal_input=torch.unsqueeze(starting_signal_input,0)
        new_point=model(starting_signal_input)
        new_point = torch.squeeze(new_point)
        generated_signal.append(new_point.detach().numpy().item())
        starting_signal = torch.cat((starting_signal[1:], new_point.reshape(1))) # craetes new 100 point input with 99 elements of old try_data and 100th.... 
        # ... element  as the new_point
    generated_signal = np.array(generated_signal)
    return generated_signal

### τυχαία δοκιμή παραγωγής σήματος και σύγκριση με το αρχικό
try_data = data[6,27, 0:600].clone().detach()
fig1 = plt.plot(try_data); plt.show()
gen_signal = generate_lfp(model, try_data[0:fc_num], 3000)
fig2 = plt.plot(gen_signal)
plt.show()

