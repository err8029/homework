#libraries (maybe needed?)
import numpy
import keras
import sklearn

#my own packages per each method
from alg1.main import Alg1
from alg2.main import Alg2
from data.export import Export

def main():
    print('starting...')

    #init objs per each method
    #---------------------------------------------------------------------

    obj_dataExport=Export()
    obj_algorithm1=Alg1()
    obj_algorithm2=Alg2()

    #exec methods here :)
    #----------------------------------------------------------------------

    #data pre processing and saving
    try:
        neg=list()
        pos=list()
        neg = obj_dataExport.read(neg,'data/rt-polarity.neg')
        pos = obj_dataExport.read(pos,'data/rt-polarity.pos')

        #check list integrity (print last x elements)
        obj_dataExport.print(neg,1)
        obj_dataExport.print(pos,2)

        #create pandas df from filtered lists
        df_neg = obj_dataExport.create_df(neg)
        df_pos = obj_dataExport.create_df(pos)

    except Exception as error:
        print('sth went wrong in data grabbing and processing')
        print(error)

    #algorithms
    obj_algorithm1.exec()
    obj_algorithm2.exec()


#define main function name
if __name__ == "__main__":
    main()
