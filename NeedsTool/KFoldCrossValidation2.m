function [test_index,test_data,train_data,test_lab,train_lab] = KFoldCrossValidation2(index,data,fold_size)

    % code which is used to shuffle all the rows of the data set
    sort_array = randperm(size(data,1));
    for i = 1: size(data,1)
        randomized_data(i,:) = data(sort_array(i),:);
        randomized_index(i,:) = index(sort_array(i),:);
    end
    % code to divide the dataset int k sub data sets.
    no_of_rows = size(data,1);

    test_index{fold_size,1} = [];
    test_data{fold_size,1} = [];
    train_data{fold_size,1} = [];
    test_lab{fold_size,1} = [];
    train_lab{fold_size,1} = [];

  block = floor(no_of_rows/fold_size);

  test_index{1} = randomized_index(1:block,2:end);
  test_data{1} = randomized_data(1:block,2:end);
  train_data{1} = randomized_data(block+1:end,2:end);
  test_lab{1} = randomized_data(1:block,1);
  train_lab{1} = randomized_data(block+1:end,1);

  for f = 2:fold_size
      test_index{f} = randomized_index((f-1)*block+1:(f)*block,2:end);
      test_data{f} = randomized_data((f-1)*block+1:(f)*block,2:end);
      train_data{f} = [randomized_data(1:(f-1)*block,2:end); randomized_data(f*block+1:end, 2:end)];
      test_lab{f} = randomized_data((f-1)*block+1:(f)*block,1);
      train_lab{f} = [randomized_data(1:(f-1)*block,1); randomized_data(f*block+1:end, 1)];
  end
end