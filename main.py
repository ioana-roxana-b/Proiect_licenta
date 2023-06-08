import additional_functions as adf
import run
import generate_dataset
if __name__ == '__main__':

   for i in range(13):
      print(i+1)
      generate_dataset.gen_config(i+1)
   #adf.delete_files()
   #run.test()



