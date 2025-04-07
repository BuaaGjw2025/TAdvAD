from data_provider.data_loader import PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader,WADISegLoader

# from data_loader import PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader,WADISegLoader

from torch.utils.data import DataLoader

data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'WADI': WADISegLoader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    
    data_set = Data(
        root_path=args.root_path,
        win_size=args.seq_len,
        # step=args.seq_len,
        flag=flag,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

if __name__ == '__main__':
    flag = 'test'
    class Args:
        def __init__(self):
            self.task_name = 'anomaly_detection'
            self.root_path = '/home/gjw/Anomaly_Detection/Attack_TCN/all_datasets/WADI'
            self.seq_len = 100
            self.data = 'WADI'
            self.batch_size = 128
            self.num_workers = 0
            self.embed = 'timeF',
            self.freq = 'h',
            self.features = 'S'
            self.target = 'OT'
    args = Args() 
    data_set, data_loader = data_provider(args, flag)
    for i, (x, y) in enumerate(data_loader):
        # print(i, x.shape, y.shape)
        if i <3:
            print(x[0])
        else:
            break