import random

def remove_duplicates(list_):
        new_list = []
        for i in list_:
            if i not in new_list:
                new_list.append(i)
        print('duplicates', len(list_)-len(new_list))
        return new_list

def make_slices(nb_data, nb_slices, zero=False):
    slice_len = nb_data//nb_slices
    if zero:
        slices = [i*slice_len for i in range(nb_slices)]
    else:
        slices = [i*slice_len for i in range(1, nb_slices)]
    slices.append(nb_data)
    return slices


def shuffle_dataset(dataset, seed=1):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    dataset_shuff = [dataset[i] for i in indices] 
    return dataset_shuff