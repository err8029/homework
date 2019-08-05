#libraries (maybe needed?)
import numpy
import pandas
import keras
import sklearn

#my own packages per each method
from alg1.main import Alg1
from alg2.main import Alg2

def main():
    print('starting...')

    #create objs per each method
    obj_algorithm1=Alg1()
    obj_algorithm2=Alg2()

    #exec methods here :)
    obj_algorithm1.exec()
    obj_algorithm2.exec()

#define main function name
if __name__ == "__main__":
    main()
