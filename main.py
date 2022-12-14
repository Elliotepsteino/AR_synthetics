import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import Lasso
import statsmodels.api as sm
import scipy.linalg as la

class CustomDataset(Dataset):

    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        X = self.dataset[:-1,item]
        Y =self.dataset[-1,item]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        sample = {"X":X,"Y":Y}
        return sample

    def __len__(self):
        return len(self.dataset[0,:])


class BasicNN(nn.Module):

    def __init__(self,inSz,outSz,hSz):
        super(BasicNN,self).__init__()
        self.fc1 = nn.Linear(inSz, hSz)
        self.fc2 = nn.Linear(hSz, outSz)

    def forward(self,x):
        x = x.float()
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class AR:

    def __init__(self,coef,init_vals,
                 data_params):
        self.coef = coef
        self.order = len(self.coef)
        self.data_params = data_params
        self.init_vals = init_vals

    def generate(self):
        c = self.data_params["Noise"]
        samples = np.zeros(self.data_params["Mhat"])
        for i in range(self.data_params["Mhat"]):
            samples[i] = np.dot(self.coef,self.init_vals)+np.random.normal()*c
            self.init_vals[1:] = self.init_vals[:-1]
            self.init_vals[0] = samples[i]
        return samples


class Data:

    def __init__(self,data_params):
        self.data_params = data_params
        self.coef = None
        self.TraincoefVec = []
        self.TestcoefVec = []

    def _get_roots(self):
        """Sample random points on the unit circle"""
        assert(self.data_params["N"]%2==0)
        half = self.data_params["N"]//2
        r= np.sqrt(np.random.uniform(0,self.data_params["r"],half))  ##We are sampling roots random, the coefficient wont have mean 0!
        theta = np.random.uniform(0,np.pi,half)
        roots_r = np.cos(theta)*r+1j*np.sin(theta)*r
        roots_l =np.cos(theta)*r-1j*np.sin(theta)*r
        roots = np.zeros(self.data_params["N"],dtype=complex)
        roots[:half]= roots_r
        roots[half:] =roots_l
        return roots

    def _new_coefs(self):
        roots = self._get_roots()
        coefs = np.polynomial.polynomial.polyfromroots(roots)
        coefs = -(coefs[:-1])[::-1]
        self.coef = np.real(coefs)

    def generate_data(self):
        Train_dat = np.zeros((self.data_params["Mhat"],self.data_params["M"]))
        Test_dat = np.zeros((self.data_params["Mhat"],self.data_params["M_tst"]))
        for i in range(self.data_params["M"]):
            self._new_coefs()
            self.TraincoefVec.append(self.coef)
            init_vals = np.random.uniform(-1,1,len(self.coef))
            ar = AR(self.coef,init_vals,self.data_params)
            Y = ar.generate()
            Train_dat[:,i] = Y

        for i in range(self.data_params["M_tst"]):
            self._new_coefs()
            self.TestcoefVec.append(self.coef)
            init_vals = np.random.uniform(-1,1,self.data_params["N"])
            ar = AR(self.coef,init_vals,self.data_params)
            Y = ar.generate()
            Test_dat[:,i] = Y


        return Train_dat,Test_dat

    def getCoefVec(self):
        return self.TraincoefVec,self.TestcoefVec


class MLP_trainer:

    def __init__(self,train_params, Train_data,Test_data):
        self.train_params = train_params

        self.trainset = CustomDataset(Train_data)
        self.testset = CustomDataset(Test_data)
        self.trainloader = DataLoader(self.trainset,batch_size=train_params["Bs"])
        self.testloader =DataLoader(self.testset,batch_size=train_params["Bs"])
        self.trainloss =[]
        self.testloss =0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Net = BasicNN(len(Train_data[:, 0]) - 1, 1, train_params["d"]).to(self.device)

    def train(self):

        optimizer = optim.SGD(self.Net.parameters(),lr =self.train_params["lr"])
        criterion = nn.MSELoss()
        Lep = []
        for i in range(self.train_params["epochs"]):
            running_Loss =0
            self.Net.zero_grad()
            for elem in self.trainloader:

                X,Y = elem["X"].to(self.device),elem["Y"].to(self.device)
                Y = Y.float()

                assert torch.isnan(torch.sum(Y))==False
                assert torch.isnan(torch.sum(X))==False
                Yhat = torch.squeeze(self.Net(X))
                assert torch.isnan(torch.sum(Yhat)) ==False
                loss = criterion(Yhat,Y)
                loss.backward()
                optimizer.step()
                #print(loss)
                running_Loss+=loss.item()

            Lep.append(running_Loss/(len(self.trainloader)))
        print(Lep)
        self.trainloss =Lep

    def eval(self):
        criterion = nn.MSELoss()
        lossVec = []
        running_Loss = 0
        self.Net.zero_grad()
        for elem in self.testloader:
            X, Y = elem["X"].to(self.device), elem["Y"].to(self.device)
            #Xnp = X.detach().numpy()
            #Ynp = Y.detach().numpy()
            Yhat = torch.squeeze(self.Net(X))
            #Yhathat = torch.zeros(Yhat.size())
            loss = criterion(Yhat, Y)

            running_Loss += loss.item()
            lossVec.append(loss.item())
        self.testloss = running_Loss/(len(self.testloader))
        self.TestlossVec = lossVec

    def get_train_loss(self):
        return self.trainloss

    def get_eval_loss(self):
        return self.testloss

    def get_eval_lossVec(self):
        return self.TestlossVec


def create_and_train(data_params,train_params):
    np.random.seed(data_params["seed"])
    torch.manual_seed(data_params["seed"])
    dat = Data(data_params)

    Train_data, Test_data = dat.generate_data()
    assert (np.isnan(np.sum(Train_data))==False)
    assert (False==np.isnan(np.sum(Test_data)))
    model = MLP_trainer(train_params,Train_data,Test_data)
    model.train()
    model.eval()

    return model


def plot_results(L_test,H_range,L,L_bound,MSEerrLasso,LossAR,data_params):
    plt.plot(H_range,L_test,label="2 Layer MLP")
    plt.plot([H_range[0],H_range[-1]],[L,L],label="Baseline constant zero prediction")
    plt.plot([H_range[0],H_range[-1]],[L_bound,L_bound],label="Lower bound on error")
    plt.plot([H_range[0],H_range[-1]],[MSEerrLasso,MSEerrLasso],label ="MSE error lasso")
    plt.plot([H_range[0],H_range[-1]],[LossAR,LossAR],label ="AR("+str(data_params["N"])+") process fitted at inference")
    plt.xlabel("Number of heads")
    plt.ylabel("MSE loss")
    plt.title("AR("+str(data_params["N"])+") transfer task with noise standard deviation \n sig = "
              +str(data_params["Noise"])+", context length Mhat = "
              +str(data_params["Mhat"])+" and \ntrain set size "+str(data_params["M"]))
    plt.legend()
    plt.show()
    pass

def fft_conv(c,phi):
    out = np.real(np.fft.fft(np.fft.fft(c)* np.fft.ifft(phi))[0:len(c)//2])
    return out

def NN_coef(X,n,params):
    N = len(X) // 2

    c = -1 / X[N - 1]
    b = X[N:]
    Bhat = -c * b
    chat = c * np.concatenate((np.array([0]), X[0:N - 1][::-1], np.array([0]), X[N:2 * N - 1][::-1]))
    sr = 0
    if params["sr"]==True:
        col1 = np.concatenate((np.array([0]), X[N:2 * N - 1]))
        row1 = np.concatenate((np.array([0]), X[0:N - 1][::-1]))
        A = c * la.toeplitz(col1, row1)
        evals = np.linalg.eigvals(A)
        sr = np.linalg.norm(evals, np.inf)

    lam = params["lambda"]
    phi = np.zeros(N)
    for i in range(n):
        philong = np.concatenate((phi,np.zeros(N)))
        mm_res =fft_conv(chat,philong)
        phi = (1-lam)*(Bhat+mm_res)+lam*phi
    y_pred = np.dot(X[N:2*N][::-1],phi)
    return phi,y_pred,sr


def experiments(params,experiment_config):
    train_params = params["train"]
    data_params = params["data"]

    if experiment_config["1"]:
        """Scaling of test loss as number of hidden dimensions increase"""
        #data_params = {"N":2,"Mhat":100,"M":10000,"M_tst":500,"Noise":0.1,"U":"unif","seed":42,"coef_sampling":"random"}
        data_params["M"]= 10000
        data_params["r"] = 0.9

        h_max = 8
        d_range = [i * i*i*4 for i in range(1, h_max + 1)]
        L_test = []
        np.random.seed(data_params["seed"])
        torch.manual_seed(data_params["seed"])
        dat = Data(data_params)

        Train_data, Test_data = dat.generate_data()
        assert (np.isnan(np.sum(Train_data)) == False)
        assert (False == np.isnan(np.sum(Test_data)))
        for d in d_range:
            train_params["d"] = d
            model = MLP_trainer(train_params, Train_data, Test_data)
            model.train()
            model.eval()
            L_test.append(model.get_eval_loss())
            print(L_test)
        dat = Data(data_params)

        tstloader = DataLoader(CustomDataset(Test_data),batch_size=train_params["Bs"])
        L = 0
        for sample in tstloader:
            Y = np.squeeze((sample["Y"]).numpy())
            L+= np.dot(Y,Y)/len(Y)
        L/=len(tstloader)

        #L= upper bound
        #L_bound = lower bound on loss
        L_bound = data_params["Noise"]*data_params["Noise"]

        #Lasso
        Xtrain = Train_data[:-1, :].T
        Ytrain = Train_data[-1, :]

        Xtest = Test_data[:-1, :].T
        Ytest = Test_data[-1, :]
        mod = Lasso(alpha=0.01).fit(Xtrain, Ytrain)
        Ypred = mod.predict(Xtest)
        MSEerrLasso = np.dot(Ypred - Ytest, Ypred - Ytest) / len(Ypred)
        base = np.dot(Ytest, Ytest) / len(Ypred)

        #AR 2 at inference
        LossAR = 0
        for i in range(len(Xtest[:,0])):
            model = sm.tsa.ARIMA(endog=Xtest[i,:], exog=None, order=(data_params["N"], 0, 0), trend="n").fit()

            coef = model.params[:-1]
            pred = np.dot(Xtest[i,:][-len(coef):][::-1],coef)
            LossAR += np.dot(pred-Ytest[i],pred-Ytest[i])

        LossAR/=len(Xtest[:,0])

        plot_results(L_test, d_range,L,L_bound,MSEerrLasso,LossAR,data_params)

    if experiment_config["2"]:
        """Use a linear regression model"""
        np.random.seed(data_params["seed"])
        torch.manual_seed(data_params["seed"])
        dat = Data(data_params)
        data_params["Mhat"] = 100
        data_params["Noise"] = 0.1
        Train_data, Test_data = dat.generate_data()
        Xtrain = Train_data[:-1, :].T
        Ytrain = Train_data[-1, :]

        Xtest = Test_data[:-1, :].T
        Ytest = Test_data[-1, :]
        mod = Lasso(alpha=0.01).fit(Xtrain,Ytrain) #All coefs are set to 0
        Ypred = mod.predict(Xtest)
        err = np.dot(Ypred-Ytest,Ypred-Ytest)/len(Ypred)
        base = np.dot(Ytest,Ytest)/len(Ypred)
        coef = mod.coef_
        intercept = mod.intercept_
        print("MSE err: ",err)
        print("Baseline: ",base)
        print("Reg coef: ",coef)
        print("Reg intercept: ",intercept)
        print(np.mean(Ytrain))

    if experiment_config["3"]:
        """Retrain each sample using a AR timeseries model (cheating, as we give the AR order)"""
        np.random.seed(data_params["seed"])
        torch.manual_seed(data_params["seed"])
        dat = Data(data_params)
        data_params["Mhat"] = 20
        data_params["Mtst"] = 30
        data_params["Noise"] = 2
        Train_data, Test_data = dat.generate_data()

        Xtest = Test_data[:-1, :].T
        Loss = 0
        Base = 0
        Ytest = Test_data[-1, :]
        for i in range(len(Xtest[:,0])):
            model = sm.tsa.ARIMA(endog=Xtest[i,:], exog=None, order=(data_params["N"], 0, 0), trend="n").fit()
            coef = model.params[:-1]
            pred = np.dot(Xtest[i,:][-len(coef):][::-1],coef)
            Loss += np.dot(pred-Ytest[i],pred-Ytest[i])
            Base+= np.dot(Ytest[i],Ytest[i])
        Loss/=len(Xtest[:,0])
        Base /=len(Xtest[:,0])
        print("AR LOSS:",Loss)
        print("Base loss: ",Base)

    if experiment_config["4"]:
        """Test the model described in the AR_synthetics_fft note"""

        N = 2
        Mhat = 2*N+1
        n = 30
        data_params_AR = copy.deepcopy(data_params)
        data_params_AR["N"] = N
        data_params_AR["Mhat"] =Mhat
        data_params_AR["M"] = 2
        data_params_AR["M_tst"] = 50
        data_params_AR["Noise"]=0
        data_params_AR["r"] = 0.8
        data_params_AR["n_iter"] = n

        np.random.seed(data_params["seed"])
        torch.manual_seed(data_params["seed"])
        dat = Data(data_params_AR)
        Train_data, Test_data = dat.generate_data()

        Xtest = Test_data[:-1, :]
        Ytest = Test_data[-1, :]

        train_coef,test_coef = dat.getCoefVec()
        train_coef = train_coef[0]

        coef_vec = []
        pred_vec = []
        sr_vec = []
        for i in range(data_params_AR["M_tst"]):
            estimated_coef, y_pred,sr = NN_coef(Xtest[:,i],n,data_params_AR)
            coef_vec.append(estimated_coef)
            pred_vec.append(y_pred)
            sr_vec.append(sr)
        pred_vec = np.array(pred_vec)
        sr_vec = np.array(sr_vec)
        abs_err = np.abs(pred_vec-Ytest)/np.abs(Ytest)
        sort_ind = np.argsort(sr_vec)
        abs_err = abs_err[sort_ind][::-1]
        sr_vec =sr_vec[sort_ind][::-1]
        plt.plot(sr_vec,np.log(abs_err))
        #plt.ylim([-5,5])
        plt.title("Forecasing from a noiseless AR("+str(data_params["N"])+") using no training data "
                                              "with \n n ="+str(data_params["n_iter"])+" model layers/iterations. Error on one step forecast \nas a function of"
                                                                                               " spectral radius of transformed data matrix")
        plt.xlabel("Spectral Radius")
        plt.ylabel("log(abs(y_pred_"+str(Mhat)+"-y_"+str(Mhat)+")/abs(y_"+str(Mhat)+"))")

        plt.show()

    if experiment_config["5"]:
        """Test that the implementation in the AR_synthetics_fft note is correct"""

        N = 2
        Mhat = 2 * N + 1
        n = 30
        data_params_AR = copy.deepcopy(data_params)
        data_params_AR["N"] = N
        data_params_AR["Mhat"] = Mhat
        data_params_AR["M"] = 2
        data_params_AR["M_tst"] = 100
        data_params_AR["Noise"] = 0
        data_params_AR["r"] = 0.8
        data_params_AR["n_iter"] = n
        np.random.seed(data_params["seed"])
        torch.manual_seed(data_params["seed"])
        dat = Data(data_params)
        Train_data, Test_data = dat.generate_data()

        Xtest = Test_data[:-1, :]
        Ytest = Test_data[-1, :]
        X = Xtest[:,0]
        N = len(X) // 2

        c = -1 / X[N - 1]
        b = X[N:]
        Bhat = -c * b
        chat = c * np.concatenate((np.array([0]), X[0:N - 1][::-1], np.array([0]), X[N:2 * N - 1][::-1]))
        sr = 0
        col1 = np.concatenate((np.array([0]), X[N:2 * N - 1]))
        assert len(col1) == N
        row1 = np.concatenate((np.array([0]), X[0:N - 1][::-1]))
        assert len(row1) == N
        A = c * la.toeplitz(col1, row1)
        evals = np.linalg.eigvals(A)
        print(evals)
        print("Spectral radius")
        print(np.linalg.norm(evals, np.inf))

        assert len(Bhat) == N
        col1 = np.concatenate((np.array([0]), X[N:2 * N - 1]))
        assert len(col1) == N
        row1 = np.concatenate((np.array([0]), X[0:N - 1][::-1]))
        assert len(row1) == N
        A = c * la.toeplitz(col1, row1)
        sr = np.linalg.norm(np.linalg.eigvals(A), ord=np.inf)
        Abar = np.zeros((2 * N, 2 * N))
        col2 = np.concatenate((np.array([0]), X[0:N - 1]))
        row2 = np.concatenate((np.array([0]), X[N:2 * N - 1][::-1]))
        D = la.toeplitz(col2, row2)
        Abar[:N, :N] = A
        Abar[N:, N:] = A
        Abar[:N, N:] = c * D
        Abar[N:, N:] = c * D
        eps = 10e-5
        assert np.linalg.norm (np.matmul(Abar,np.concatenate((Bhat, np.zeros(N))))[:N]-fft_conv(chat,np.concatenate((Bhat, np.zeros(N)))))<eps
        assert np.linalg.norm(np.matmul(A,Bhat) -fft_conv(chat,np.concatenate((Bhat, np.zeros(N)))))<eps
        assert np.all(chat == Abar[0, :])
        assert len(chat) == 2 * N
        print("All tests passed")

def main():
    #N = Degree of AR process
    #Mhat =  samples per datapoint
    #M = number of datapoints in train set
    #Mhat = number of dataponts in test set
    #Noise = Include noise term in AR process
    #sig = noise variance
    #U = sampling distribution for initial points for each sample
    #seed = RNG initial seed
    #Noise = variance of noise
    #sr = get spectral radius
    data_params = {"N":2,"Mhat":50,"M":500,"M_tst":500,"Noise":0.1,
                   "U":"unif","seed":42,"coef_sampling":"random","r":1,"n_iter":50,"sr":True,"lambda":0.2}

    #lr = learning rate
    #h = model heads
    #d = hidden size
    #Bs = batch size
    train_params = {"lr":0.00001,"h":5,"d":5,"Bs":50,"epochs":100}
    experiment_config = {"1":True,"2":False,"3":False,"4":False,"5":False}
    params = {"train":train_params,"data":data_params}
    experiments(params,experiment_config)
    # Experiment 1: Run Lasso,AR fitting, and basic MLP on the timeseries transfer task from transfer time series note
    # Experiment 2: Only run Lasso
    # Experiment 3: Only run AR fitting
    # Experiment 4: Run an experiment from the AR_synthetics_fft note
    # Experiment 5: Confirm that the calculations in experiment 4 are correct

if __name__ == '__main__':
    main()

