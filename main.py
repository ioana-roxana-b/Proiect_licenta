import additional_functions as adf
import run
import time
import generate_dataset
if __name__ == '__main__':

    adf.delete_files('Results')
    start_time = time.time()
    run.test()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Elapsed time: {} seconds".format(elapsed_time))



