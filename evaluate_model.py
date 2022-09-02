import numpy as np
import torch
import time
from thop import profile


def calculate_params_flops(models,input,print=False):
    device = 'cpu'
    input = input.to(device)

    ans_arr = np.zeros([len(models),3])#model->param-flops-param
    
    for i in range(len(models)):
        model = models[i].to(device)
        
        #if model.is_sequentioal: #test
        #    model.reset_HXs(input.shape[0],4,13,True)
        macs, params = profile(model, inputs=(input,))
        ans_arr[i,0] = (params/1000000.0)
        ans_arr[i,1] = (macs/1000000.0)
        ans_arr[i,2] = count_parameters(model)
    
    if print:
        print('Total params: %.2fM' % (params/1000000.0))
        print('Total flops: %.2fM' % (macs/1000000.0))
    else:
        return ans_arr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def calulate_inf_times___(models:list,input,device='cpu'):
    
    c,h,w = input.shape
    batch_max = 5
    count_of_inf = 10
    count_of_skips = 5
    inf_times = np.zeros([len(models),count_of_inf,batch_max])#model-time-batchs_time
    ans_arr = np.zeros([len(models),1])#model-time
    
    if(isinstance(models,list)):
        for model in models:
            torch.cuda.empty_cache()
            model = model.to(device)
            #print(next(model.parameters()).device)
            if model.is_sequentioal: #test
                model.reset_HXs(input.shape[0],4,13,True)
        for i in range(0,count_of_inf):
            for b_size in range(batch_max):
                inp = torch.rand((b_size+1,c,h,w),dtype=torch.float32).to(device)
                #input = input.to(device)
                for j in range(len(models)):
                    start_time = time.time()
                    x = models[j](inp)
                    inf_time = time.time()-start_time
                    inf_times[j,i,b_size]=inf_time/(b_size+1)
                    
        inf_times = np.mean(inf_times,axis=2)
        for k in range(len(models)):
            #b_inf = 0
            #for b_size in range(batch_max):   
            ans_arr[k,0]=np.mean(inf_times[k,count_of_skips:])
    
    return ans_arr

@torch.no_grad()
def calulate_inf_times(models:list,input,device='cpu',n_inf=50):
    
    count_of_inf = n_inf
    count_of_skips = 5
    inf_times = np.zeros([len(models),count_of_inf])#model-time
    ans_arr = np.zeros([len(models),1])#model-time
    

    input = input.to(device)

    if(isinstance(models,list)):
        for model in models:
            torch.cuda.empty_cache()
            model = model.to(device)
            #print(next(model.parameters()).device)
            #if model.is_sequentioal: #test
            #    model.reset_HXs(input.shape[0],4,13,True)
        for i in range(0,count_of_inf):
            for j in range(len(models)):
                start_time = time.time()
                x= models[j](input)
                inf_time = time.time()-start_time
                inf_times[j,i]=inf_time
    
        for k in range(len(models)):
            ans_arr[k,0]=np.mean(inf_times[k,count_of_skips:])
    
    return ans_arr


def evaluate_model(models:list,input=torch.randn(1, 1, 6, 128, 416),print_res=True,n_inf=50):

    result = np.ones((len(models),6),dtype=np.float)*999

    inf_times_cpu = calulate_inf_times(models,input,'cpu',n_inf=n_inf)

    inf_times_gpu = calulate_inf_times(models,input,'cuda:0',n_inf=n_inf)
    param_flops = calculate_params_flops(models,input)

    for i in range(len(models)):
        result[i,0]=i+1
        result[i,1]=inf_times_cpu[i,0]
        result[i,2]=inf_times_gpu[i,0]
        result[i,3]=param_flops[i,0]#par
        result[i,4]=param_flops[i,1]#flops
        result[i,5]=param_flops[i,2]#par

    if print_res:
        print('*'*85)
        print('  id\t |inf_cpu\t|inf_gpu\t|params\t\t|MACs\t\t|params')
        print('-'*85) 
        for i in range(result.shape[0]):
            #print('{int(result[i,0])}\t {result[i,1]}\t{result[i,2]}\t{result[i,3]}\t{result[i,4]}\t{int(result[i,5])}\t')\
            print("  {}\t {:.5f}\t{:.5f}\t\t{:.2f}M\t\t{:.2f}MF\t{}".format(int(result[i,0]),result[i,1],result[i,2],result[i,3],result[i,4],int(result[i,5])))
            print('-'*85)
    else:
        return result        



'''
from torchvision.models import resnet18,resnet34
model1 = resnet18()
model2 = resnet34()
evaluate_model([model1,model2])
'''