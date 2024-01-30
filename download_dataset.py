import opendatasets as od

def download_dataset():
    dataset = 'https://www.kaggle.com/c/career-con-2019/'
    od.download(dataset)


if __name__ == "__main__":
    download_dataset()