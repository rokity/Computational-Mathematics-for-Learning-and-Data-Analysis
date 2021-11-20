from neural_network import NeuralNetwork
from parser import Monks_parser
from utility import set_style_plot
from utility import *
from early_stopping import *
from kernel_initialization import *

set_style_plot()

DIR_DATA = "./data/"
TR_FILE = 'monks-1.train'
TS_FILE = 'monks-1.test'


path_tr = DIR_DATA + TR_FILE
path_ts = DIR_DATA + TS_FILE
path_loss = 'out/err_monk1nr.png'
path_acc = 'out/acc_monk1nr.png'
path_result_model = 'out/results_1_nr.csv'

dim_in = 6
one_hot = 17
dim_out = 1

parser = Monks_parser(path_tr, path_ts)

X_train, Y_train, X_test, Y_test = parser.parse(dim_in, dim_out, one_hot)

#Y_train = change_output_value(Y_train, 0, -1)
#Y_test = change_output_value(Y_test, 0, -1)

#X_train, Y_train, X_val, Y_val = train_test_split(X_train, Y_train, test_size=0.25)

dim_in = one_hot
dim_hid = 4

model = NeuralNetwork('mse', 'accuracy')

model.add_layer(dim_hid, input_dim=dim_in, activation='tanh', kernel_initialization=RandomUniformInitialization(-0.5, 0.5))
model.add_layer(dim_out, activation='tanh', kernel_initialization=RandomUniformInitialization(-0.5, 0.5))

model.compile()
model.fit(
    X_train, Y_train, 500, X_train.shape[0], ts=(X_test, Y_test),
    verbose=True, tol=1e-2
)

err_tr, acc_tr = model.evaluate(X_train, Y_train)
err_ts, acc_ts = model.evaluate(X_test, Y_test)
acc_tr = acc_tr*100
acc_ts = acc_ts*100
errors = [err_tr, err_ts]
accuracy = [acc_tr, acc_ts]

res = {
    'Error': errors,
    'Accuracy': accuracy,
}

print(res)
write_results(res, model, path_loss, path_acc, path_result_model, False, True, True)
