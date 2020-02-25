from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import sys
def board2csv(current_directory, file_names, algo):
    #sys.path.append(current_directory)
    print(current_directory)
    for i, file_name in enumerate(file_names):
        ea = event_accumulator.EventAccumulator(current_directory+'/logs/'+file_name)
        ea.Reload()
        tags = ea.Tags()
        for j, tag in enumerate(tags['scalars']):
            pd.DataFrame(ea.Scalars(tag)).to_csv(current_directory+'/csv_log/'+str(i+1)+'_'+str(j+1)+'_train.csv')
