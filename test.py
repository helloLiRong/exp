import os  
  
mystr=os.popen("python train_np.py")  #popen与system可以执行指令,popen可以接受返回对象  
mystr=mystr.read() #读取输出  
