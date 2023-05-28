import additional_functions as adf
import grad_boosting
import generate_dataset
import lgbm

if __name__ == '__main__':

   adf.delete_files()
   adf.test_rfe()
   """""
   train, test = adf.read_data(1)
   data = adf.read_data_once(1)
   grad_boosting.gradient_boosting(config=1, train_data_df=train, test_data_df=test,
                                                           data_df=data,
                                                           shuffle=True, pc=False,
                                   scal=True, minmax=False, lasso=False,rfe=True)
    """
   """"Acuratete 100% ?????""
   train, test = adf.read_data(4)
   data = adf.read_data_once(4)
   lgbm.lightgbm(config=4, train_data_df=train, test_data_df=test, data_df=data, pc=False, scal=True, minmax=False, lasso=True, rfe=False)
   """

