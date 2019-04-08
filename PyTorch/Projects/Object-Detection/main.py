try:
    import dataLoader.load_data as ld
except ImportError:
    print("Exception")

cifar_data = ld.Dataset()
input_features, labels = cifar_data.load_data(batch_num=5)
print("Number of features: {}, Number of Labels: {}".format(len(input_features), len(labels)))