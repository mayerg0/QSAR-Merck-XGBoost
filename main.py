import sys
from train_test import *

def train_test_model(model, dataset, kfold):
    training_set_path = get_file_path(i)
    print('Reading file: ', training_set_path)
    training_df = pd.read_csv(training_set_path) 
    if kfold == 0:
        print('train_test_split')
        train_test = train_test_split(training_df, 0.8)
        train = train_test[0]
        test = train_test[1]
        print('training model')
        trained_model = train_model(train, model)
        print('testing model')
        test = predict_test(test, trained_model)
        print('evaluating model')
        evaluation_results = evaluation(test, 'Act', 'predictions')
        print(evaluation_results)
        return [i, evaluation_results, test[['Act','predictions']]]
    elif kfold == 1:
        relevant_columns = list(training_df.columns)
        relevant_columns.remove('MOLECULE')
        relevant_columns.remove('Act')
        kfold = KFold(n_splits = 5, random_state=123)
        results_kfold = cross_val_score(model, X = training_df[relevant_columns], y = training_df['Act'], cv=kfold)
        return results_kfold


def main():
    # default command line arguments
    if len(sys.argv) == 1:
        dataset = 'all'
        model = 'xgb'
        
    dataset = sys.argv[1]
    model = sys.argv[2]

    if model == 'dnn':
        model = merck_dnn_model()
    elif model == 'rf':
        model = random_forest_model()
    elif model == 'xgb':
        model = xgb_model()
        
    if dataset == 'all': 
        results = []
        for i in range(1,16):
            results.append(train_Test_model(model, dataset, kfold = 0))
        results_dataframe = pd.DataFrame(results, columns = ['index','results','data'])
        results_dataframe['r2']=results_dataframe['results'].apply(lambda x: x['r2'])
        results_dataframe.to_csv('results.csv')                     
           
    else:
        results = train_Test_model(model, dataset, kfold = 1)
        print(results)
        
    
if __name__ == "__main__":
    main()