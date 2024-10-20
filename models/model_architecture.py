# -*- coding: utf-8 -*-

import torch
from torch.nn import Linear, Sigmoid, Module, Sequential, ReLU, Tanh

class race(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 2
        super(race, self).__init__()        
        # Fully connected layers
        self.fc_1 = Sequential(Linear(self.input_dim, self.hidden_dim*2),Tanh())
        self.fc_2 = Sequential(Linear(self.hidden_dim*2, self.hidden_dim*4),Tanh())
        self.fc_3 = Sequential(Linear(self.hidden_dim*4, self.hidden_dim*2),Tanh())
        self.fc_4 = Sequential(Linear(self.hidden_dim*2, self.output_dim))       
    def forward(self,x):
        out_fc1 = self.fc_1(x)
        out_fc2 = self.fc_2(out_fc1)
        out_fc3 = self.fc_3(out_fc2)
        out = self.fc_4(out_fc3)
        return out
        
class plant_net_v1(Module):
    def __init__(self):
        
        self.hidden_dim = 3
        self.output_dim = 1
        super(plant_net_v1, self).__init__()        
        # Fully connected layers
        self.fc_1 = Sequential(Linear(3, self.hidden_dim),Sigmoid())
        self.fc_2 = Sequential(Linear(self.hidden_dim, self.output_dim))
        
    def forward(self,x):
        
        out_fc1 = self.fc_1(x)
        out = self.fc_2(out_fc1)
        
        return out
    
class plant_net_xy(Module):
    def __init__(self):
        
        self.input_dim = 3
        self.hidden_dim = 3
        self.output_dim = 1
        super(plant_net_xy, self).__init__()        
        # Fully connected layers
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4x = Sequential(Linear(self.hidden_dim, self.output_dim))

        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4y = Sequential(Linear(self.hidden_dim, self.output_dim))        

    def forward(self,x,y):
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        out_4x = self.fc_4x(out_fc3x)
        
        out_fc1y = self.fc_1y(y)
        out_fc2y = self.fc_2y(out_fc1y)
        out_fc3y = self.fc_3y(out_fc2y)
        out_4y = self.fc_4y(out_fc3y)
        
        out = torch.cat((out_4x,out_4y),dim=0)    
        return out

class plant_net_xandy(Module):
    def __init__(self):
        self.input_dim = 6
        self.hidden_dim = 12
        self.output_dim = 2
        super(plant_net_xandy, self).__init__()        
        # Fully connected layers
        self.fc_1 = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
#        self.fc_2x = Sequential(Linear(self.hidden_dim),Sigmoid())
#        self.fc_3x = Sequential(Linear(3, self.hidden_dim),Sigmoid())
        self.fc_2 = Sequential(Linear(self.hidden_dim, self.output_dim))

    def forward(self,x):
        out_fc1 = self.fc_1(x)
        out = self.fc_2(out_fc1)
        return out

class plant_net_xandy_v2(Module):
    def __init__(self):
        self.input_dim = 6
        self.hidden_dim = 12
        self.output_dim = 2
        super(plant_net_xandy_v2, self).__init__()        
        # Fully connected layers
        self.fc_1 = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2 = Sequential(Linear(self.hidden_dim,self.hidden_dim),Sigmoid())
        self.fc_3 = Sequential(Linear(self.hidden_dim, 6),Sigmoid())
        self.fc_4 = Sequential(Linear(6, self.output_dim))

    def forward(self,x):
        out_fc1 = self.fc_1(x)
        out_fc2 = self.fc_2(out_fc1)
        out_fc3 = self.fc_3(out_fc2)
        out = self.fc_4(out_fc3)
        
        return out
    
class plant_net_v2(Module):
    def __init__(self):
        
        self.hidden_dim = 3
        self.output_dim = 1
        super(plant_net_v2, self).__init__()        
        # Fully connected layers
        self.fc_1 = Sequential(Linear(3, self.hidden_dim),Sigmoid())
        self.fc_2 = Sequential(Linear(3, self.hidden_dim),Sigmoid())
        self.fc_3 = Sequential(Linear(3, self.hidden_dim),Sigmoid())
        self.fc_4 = Sequential(Linear(self.hidden_dim, self.output_dim))
        
    def forward(self,x):
        
        out_fc1 = self.fc_1(x)
        out_fc2 = self.fc_2(out_fc1)
        out_fc3 = self.fc_3(out_fc2)
        out = self.fc_4(out_fc3)
        
        return out
    
class ctrlr_net_xy(Module):
    def __init__(self):
        self.input_dim = 5
        self.hidden_dim = 4
        self.output_dim = 1
        super(ctrlr_net_xy, self).__init__()        
        # Fully connected layers
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        
    def forward(self,x,y):
        
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc1y = self.fc_1y(y)
        out_fc2y = self.fc_2y(out_fc1y)
        out = torch.cat((out_fc2x,out_fc2y),dim=0)
        return out

class ctrlr_net_b_xy(Module):
    def __init__(self):
        self.input_dim = 3
        self.hidden_dim = 5
        self.output_dim = 1
        super(ctrlr_net_b_xy, self).__init__()        
        # Fully connected layers
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.input_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.input_dim, self.output_dim))
        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.input_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.input_dim, self.output_dim))
        
    def forward(self,x,y):
        
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        out_fc1y = self.fc_1y(y)
        out_fc2y = self.fc_2y(out_fc1y)
        out_fc3y = self.fc_3y(out_fc2y)
        out = torch.cat((out_fc3x,out_fc3y),dim=0)
        return out
    
class ctrlr_net_b_seq_xy(Module):
    def __init__(self):
        self.input_dim = 3
        self.hidden_dim = 4
        self.output_dim = 4
        super(ctrlr_net_b_seq_xy, self).__init__()        
        # Fully connected layers
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_5x = Sequential(Linear(self.hidden_dim, self.output_dim))
        
        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_5y = Sequential(Linear(self.hidden_dim, self.output_dim))
        
    def forward(self,x,y):
        
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        out_fc4x = self.fc_4x(out_fc3x)
        out_fc5x = self.fc_5x(out_fc4x)
        
        out_fc1y = self.fc_1y(y)
        out_fc2y = self.fc_2y(out_fc1y)
        out_fc3y = self.fc_3y(out_fc2y)
        out_fc4y = self.fc_4y(out_fc3y)
        out_fc5y = self.fc_5y(out_fc4y)
        
        out = torch.cat((out_fc5x,out_fc5y),dim=0)
        return out

class ctrlr_net_lr(Module):
    def __init__(self):
        self.input_dim = 5
        self.hidden_dim = 4
        self.output_dim = 1
        super(ctrlr_net_lr, self).__init__()        
        # LEFT HS - gets right input, gives right output
        self.fc_1xl = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2xl = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
#        self.fc_3xl = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3xl = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        self.fc_1yl = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2yl = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
#        self.fc_3yl = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3yl = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        # RIGHT HS - gets left input, gives left output
        self.fc_1xr = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2xr = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
#        self.fc_3xr = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3xr = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        self.fc_1yr = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2yr = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
#        self.fc_3yr = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3yr = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        
    def forward(self,xl,yl,xr,yr):
        # first FC
        out_fc1xr = self.fc_1xr(xl)
        out_fc2xr = self.fc_2xr(out_fc1xr)
#        out_fc3xr = self.fc_3xr(out_fc2xr)
        out_fc3xr = self.fc_3xr(out_fc2xr)
        
        out_fc1yr = self.fc_1yr(yl)    
        out_fc2yr = self.fc_2yr(out_fc1yr)
#        out_fc3yr = self.fc_3yr(out_fc2yr)
        out_fc3yr = self.fc_3yr(out_fc2yr)
        
        out_fc1xl = self.fc_1xl(xr)
        out_fc2xl = self.fc_2xl(out_fc1xl)
#        out_fc3xl = self.fc_3xl(out_fc2xl)        
        out_fc3xl = self.fc_3xl(out_fc2xl)        
        
        out_fc1yl = self.fc_1yl(yr)
        out_fc2yl = self.fc_2yl(out_fc1yl)
#        out_fc3yl = self.fc_3yl(out_fc2yl)
        out_fc3yl = self.fc_3yl(out_fc2yl)
        out = torch.cat((out_fc3xr,out_fc3yr,out_fc3xl,out_fc3yl),dim=0)
        return out
        # combining activity from x of right and left
        # v3
#        comb_fc2xl = out_fc2xl + out_fc2xr + out_fc2yr
#        comb_fc2xr = out_fc2xr + out_fc2xl + out_fc2yl
#        comb_fc2yl = out_fc2yl + out_fc2xr + out_fc2yr
#        comb_fc2yr = out_fc2yr + out_fc2xl + out_fc2yl
        # v1
#        comb_fc2x = out_fc2xl + out_fc2xr
#        comb_fc2y = out_fc2yl + out_fc2yr
        # third FC - recieves from both sides
#        # RIGHT HS - gets left input, gives left output
#        out_fc1xr = self.fc_1xr(xl)
#        out_fc2xr = self.fc_2xr(out_fc1xr)
#        comb_fc2xr = out_fc2xr+0.1*(out_fc2xl)
#        out_fc3xr = self.fc_3xr(comb_fc2xr)
#        out_fc3xr = torch.reshape(out_fc3xr,(1,))
#        
#        out_fc1yr = self.fc_1yr(yl)
#        out_fc2yr = self.fc_2yr(out_fc1yr)
#        comb_fc2yr = out_fc2yr+0.1*(out_fc2yl)
#        out_fc3yr = self.fc_3yr(comb_fc2yr)
#        out_fc3yr = torch.reshape(out_fc3yr,(1,))
#        
#        # LEFT HS - gets right input, gives right output
#        out_fc1xl = self.fc_1xl(xr)
#        out_fc2xl = self.fc_2xl(out_fc1xl)
#        comb_fc2xl = out_fc2xl+0.05*(out_fc2xr)
#        out_fc3xl = self.fc_3xl(comb_fc2xl)
#        
#        out_fc1yl = self.fc_1yl(yr)
#        out_fc2yl = self.fc_2yl(out_fc1yl)
#        comb_fc2yl = out_fc2yl+0.05*(out_fc2yr)
#        out_fc3yl = self.fc_3yl(comb_fc2yl)
        


class eied_ctrlr(Module):
    def __init__(self):
        self.input_dim = 5
        self.hidden_dim = 4
        self.output_dim = 1
        super(eied_ctrlr, self).__init__()        
        # LEFT HS - gets right and left input, gives right and left output
        self.fc_1lxl = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2lxl = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3lxl = Sequential(Linear(self.hidden_dim, 2))#,Tanh())
        self.fc_1lyl = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2lyl = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3lyl = Sequential(Linear(self.hidden_dim, 2))#,Tanh())
        # LEFT HS - gets right and left input, gives right and left output
        self.fc_1lxr = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2lxr = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3lxr = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        self.fc_1lyr = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2lyr = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3lyr = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        # RIGHT HS - gets left input, gives left output
        self.fc_1rxl = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2rxl = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3rxl = Sequential(Linear(self.hidden_dim, 2))#,Tanh())
        self.fc_1ryl = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2ryl = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3ryl = Sequential(Linear(self.hidden_dim, 2))#,Tanh())
        self.fc_4xl = Sequential(Linear(self.hidden_dim,self.output_dim))
        self.fc_4yl = Sequential(Linear(self.hidden_dim,self.output_dim))
    def forward(self,xl,yl,xr,yr):
        # LEFT HS - left input and left output
        out_fc1lxl = self.fc_1lxl(xl)
        out_fc2lxl = self.fc_2lxl(out_fc1lxl)
        out_fc3lxl = self.fc_3lxl(out_fc2lxl)
        out_fc1lyl = self.fc_1lyl(yl)
        out_fc2lyl = self.fc_2lyl(out_fc1lyl)
        out_fc3lyl = self.fc_3lyl(out_fc2lyl)
        # RIGHT HS - left input and right output
        out_fc1rxl = self.fc_1rxl(xl)
        out_fc2rxl = self.fc_2rxl(out_fc1rxl)
        out_fc3rxl = self.fc_3rxl(out_fc2rxl)
        out_fc1ryl = self.fc_1ryl(yl)
        out_fc2ryl = self.fc_2ryl(out_fc1ryl)
        out_fc3ryl = self.fc_3ryl(out_fc2ryl)
        # LEFT HS - right input and right output
        out_fc1lxr = self.fc_1lxr(xr)
        out_fc2lxr = self.fc_2lxr(out_fc1lxr)
        out_fc3lxr = self.fc_3lxr(out_fc2lxr)
        out_fc1lyr = self.fc_1lyr(yl)
        out_fc2lyr = self.fc_2lyr(out_fc1lyr)
        out_fc3lyr = self.fc_3lyr(out_fc2lyr)
        # combinging RIGHT and LEFT HS output to get left output
        out_xl = torch.cat((out_fc3lxl,out_fc3rxl),dim=0)
        out_yl = torch.cat((out_fc3lyl,out_fc3ryl),dim=0)
        out_fc4xl = self.fc_4xl(out_xl)
        out_fc4yl = self.fc_4yl(out_yl)
        out = torch.cat((out_fc3lxr,out_fc3lyr,out_fc4xl,out_fc4yl),dim=0)
        return out

class ctrlr_net_ei(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 1
        super(ctrlr_net_ei, self).__init__()        
        # LEFT HS - gets right input, gives right output
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_5x = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_5y = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        
    def forward(self,x,y):
   
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        out_fc4x = self.fc_4x(out_fc3x)
        out_fc5x = self.fc_5x(out_fc4x)
        out_fc1y = self.fc_1y(y)    
        out_fc2y = self.fc_2y(out_fc1y)
        out_fc3y = self.fc_3y(out_fc2y)
        out_fc4y = self.fc_4y(out_fc3y)
        out_fc5y = self.fc_5y(out_fc4y)        
        out = torch.cat((out_fc5x,out_fc5y),dim=0)
        return out

class ei_ctrlr(Module):
    def __init__(self):
        self.input_dim = 18
        self.hidden_dim = 18
        self.output_dim = 1
        super(ei_ctrlr, self).__init__()        
        self.fc_1 = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2 = Sequential(Linear(self.hidden_dim, 16),Sigmoid())
        self.fc_3 = Sequential(Linear(16, 14),Sigmoid())
        self.fc_4l = Sequential(Linear(14, 12),Sigmoid())
        self.fc_4r = Sequential(Linear(14, 12),Sigmoid())
        self.fc_5l = Sequential(Linear(12, 10),Sigmoid())
        self.fc_5r = Sequential(Linear(12, 10),Sigmoid())
        self.fc_6lx = Sequential(Linear(10, 4),Sigmoid())
        self.fc_6ly = Sequential(Linear(10, 4),Sigmoid())
        self.fc_6rx = Sequential(Linear(10, 4),Sigmoid())
        self.fc_6ry = Sequential(Linear(10, 4),Sigmoid())
        self.fc_7lx = Sequential(Linear(4, 2),Sigmoid())
        self.fc_7ly = Sequential(Linear(4, 2),Sigmoid())
        self.fc_7rx = Sequential(Linear(4, 2),Sigmoid())
        self.fc_7ry = Sequential(Linear(4, 2),Sigmoid())
        self.fc_8lx = Sequential(Linear(2, self.output_dim))
        self.fc_8ly = Sequential(Linear(2, self.output_dim))
        self.fc_8rx = Sequential(Linear(2, self.output_dim))
        self.fc_8ry = Sequential(Linear(2, self.output_dim))
        
    def forward(self,inp):
        # first FC
        out_fc1 = self.fc_1(inp)
        out_fc2 = self.fc_2(out_fc1)    
        out_fc3 = self.fc_3(out_fc2)
        out_fc4l = self.fc_4l(out_fc3)
        out_fc4r = self.fc_4r(out_fc3)
        out_fc5l = self.fc_5l(out_fc4l)
        out_fc5r = self.fc_5r(out_fc4r)
        out_fc6lx = self.fc_6lx(out_fc5l)
        out_fc6ly = self.fc_6ly(out_fc5l)
        out_fc6rx = self.fc_6rx(out_fc5r)
        out_fc6ry = self.fc_6ry(out_fc5r)
        out_fc7lx = self.fc_7lx(out_fc6lx)
        out_fc7ly = self.fc_7ly(out_fc6ly)
        out_fc7rx = self.fc_7rx(out_fc6rx)
        out_fc7ry = self.fc_7ry(out_fc6ry)
        out_lx = self.fc_8lx(out_fc7lx)
        out_ly = self.fc_8ly(out_fc7ly)
        out_rx = self.fc_8rx(out_fc7rx)
        out_ry = self.fc_8ry(out_fc7ry)
#        out = self.fc_8(out_fc7)
        out = torch.cat((out_lx,out_ly,out_rx,out_ry),dim=0)
        return out
        
class ctrlr_m1(Module):
    def __init__(self):
        self.input_dim = 6
        self.hidden_dim = 4
        self.output_dim = 1
        super(ctrlr_m1, self).__init__()        
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        # self.fc_3x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        # self.fc_4x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.output_dim))

        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        # self.fc_3y = Sequential(Linear(self.hidden_dim*50, self.hidden_dim*25),Sigmoid())
        # self.fc_4y = Sequential(Linear(self.hidden_dim*25, self.hidden_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.hidden_dim, self.output_dim))
        
    def forward(self,x,y):
        # first FC
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        # out_fc4x = self.fc_4x(out_fc3x)
        # out_fc5x = self.fc_5x(out_fc4x)
        
        out_fc1y = self.fc_1y(y)    
        out_fc2y = self.fc_2y(out_fc1y)
        out_fc3y = self.fc_3y(out_fc2y)
        # out_fc4y = self.fc_4y(out_fc3y)
        # out_fc5y = self.fc_5y(out_fc4y)

        out = torch.cat((out_fc3x,out_fc3y),dim=0)
        return out

class ctrlr_ppc(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 1
        super(ctrlr_ppc, self).__init__()        
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
#        self.fc_3x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.output_dim))

        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
#        self.fc_3y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.hidden_dim, self.output_dim))
        
    def forward(self,x,y):
        # first FC
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
#        out_fc4x = self.fc_4x(out_fc3x)
        out_fc1y = self.fc_1y(y)    
        out_fc2y = self.fc_2y(out_fc1y)
        out_fc3y = self.fc_3y(out_fc2y)
#        out_fc4y = self.fc_4y(out_fc3y)
        out = torch.cat((out_fc3x,out_fc3y),dim=0)
        return out
    
class ppc_fused(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 1
        super(ppc_fused, self).__init__()        
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4x = Sequential(Linear(self.hidden_dim, self.output_dim))
        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4y = Sequential(Linear(self.hidden_dim, self.output_dim))
        
        self.fc_1d = Sequential(Linear(self.input_dim, self.hidden_dim*8),Sigmoid())
        self.fc_2d = Sequential(Linear(self.hidden_dim*8, self.hidden_dim*32),Sigmoid())
        self.fc_3d = Sequential(Linear(self.hidden_dim*32, self.hidden_dim*8),Sigmoid())
        self.fc_4d = Sequential(Linear(self.hidden_dim*8, self.hidden_dim),Sigmoid())
        self.fc_5d = Sequential(Linear(self.hidden_dim, self.output_dim))
        
        self.fc_1t = Sequential(Linear(self.input_dim, self.hidden_dim*8),Sigmoid())
        self.fc_2t = Sequential(Linear(self.hidden_dim*8, self.hidden_dim*32),Sigmoid())
        self.fc_3t = Sequential(Linear(self.hidden_dim*32, self.hidden_dim*8),Sigmoid())
        self.fc_4t = Sequential(Linear(self.hidden_dim*8, self.hidden_dim),Sigmoid())
        self.fc_5t= Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        
    def forward(self,x,y,d,t):
        # first FC
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        out_fc4x = self.fc_4x(out_fc3x)
        out_fc4x = torch.reshape((out_fc4x),(1,1))
        out_fc1y = self.fc_1y(y)    
        out_fc2y = self.fc_2y(out_fc1y)
        out_fc3y = self.fc_3y(out_fc2y)
        out_fc4y = self.fc_4y(out_fc3y)
        out_fc4y = torch.reshape((out_fc4y),(1,1))
        
        out_fc1d = self.fc_1d(d)
        out_fc2d = self.fc_2d(out_fc1d)
        out_fc3d = self.fc_3d(out_fc2d)
        out_fc4d = self.fc_4d(out_fc3d)
        out_d = self.fc_5d(out_fc4d)
        out_d = torch.reshape((out_d),(1,1))
        
        out_fc1t = self.fc_1t(t)
        out_fc2t = self.fc_2t(out_fc1t)
        out_fc3t = self.fc_3t(out_fc2t)
        out_fc4t = self.fc_4t(out_fc3t)
        out_t = self.fc_5t(out_fc4t)
        out_t = torch.reshape((out_t),(1,1))
        out = torch.cat((out_fc4x,out_fc4y,out_d,out_t),dim=0)
        return out

class rppc(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 1
        super(rppc, self).__init__()        
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4x = Sequential(Linear(self.hidden_dim, self.output_dim))
        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4y = Sequential(Linear(self.hidden_dim, self.output_dim))
        
        self.fc_1d = Sequential(Linear(self.input_dim, self.hidden_dim*8),Sigmoid())
        self.fc_2d = Sequential(Linear(self.hidden_dim*8, self.hidden_dim*32),Sigmoid())
        self.fc_3d = Sequential(Linear(self.hidden_dim*32, self.hidden_dim*8),Sigmoid())
        self.fc_4d = Sequential(Linear(self.hidden_dim*8, self.hidden_dim),Sigmoid())
        self.fc_5d = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        
    def forward(self,x,y,d,b):
        # first FC
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        # outfc3x = out_fc3x*0.45
        # out_fc4x = self.fc_4x(outfc3x)
        outfc3x = out_fc3x*b
        out_fc4x = self.fc_4x(outfc3x)
        out_fc4x = torch.reshape((out_fc4x),(1,1))
        
        out_fc1y = self.fc_1y(y)    
        out_fc2y = self.fc_2y(out_fc1y)
        out_fc3y = self.fc_3y(out_fc2y)
        # outfc3y = out_fc3y*0.45
        # out_fc4y = self.fc_4y(outfc3y)
        outfc3y = out_fc3y*b
        out_fc4y = self.fc_4y(outfc3y)
        out_fc4y = torch.reshape((out_fc4y),(1,1))
        
        out_fc1d = self.fc_1d(d)
        out_fc2d = self.fc_2d(out_fc1d)
        out_fc3d = self.fc_3d(out_fc2d)
        # outfc3d = out_fc3d*0.45
        # out_fc4d = self.fc_4d(outfc3d)
        outfc3d = out_fc3d*b
        out_fc4d = self.fc_4d(outfc3d)
        out_d = self.fc_5d(out_fc4d)
        out_d = torch.reshape((out_d),(1,1))
        
        out = torch.cat((out_fc4x,out_fc4y,out_d),dim=0)
        return out
    
class lppc(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 1
        super(lppc, self).__init__()        
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4x = Sequential(Linear(self.hidden_dim, self.output_dim))
        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4y = Sequential(Linear(self.hidden_dim, self.output_dim))
        
        self.fc_1d = Sequential(Linear(self.input_dim, self.hidden_dim*8),Sigmoid())
        self.fc_2d = Sequential(Linear(self.hidden_dim*8, self.hidden_dim*32),Sigmoid())
        self.fc_3d = Sequential(Linear(self.hidden_dim*32, self.hidden_dim*8),Sigmoid())
        self.fc_4d = Sequential(Linear(self.hidden_dim*8, self.hidden_dim),Sigmoid())
        self.fc_5d = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        
    def forward(self,x,y,t,b):
        # first FC
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        # if b == 1:
        #     outfc3x = out_fc3x*0.75
        #     out_fc4x = self.fc_4x(outfc3x)
        # else:
        outfc3x = out_fc3x*b
        out_fc4x = self.fc_4x(outfc3x)
        out_fc4x = torch.reshape((out_fc4x),(1,1))
        
        out_fc1y = self.fc_1y(y)    
        out_fc2y = self.fc_2y(out_fc1y)
        out_fc3y = self.fc_3y(out_fc2y)
        # if b == 1:
        #     outfc3y = out_fc3y*0.75
        #     out_fc4y = self.fc_4y(outfc3y)
        # else:
        outfc3y = out_fc3y*b
        out_fc4y = self.fc_4y(outfc3y)
        out_fc4y = torch.reshape((out_fc4y),(1,1))
        
        out_fc1d = self.fc_1d(t)
        outfc1d = out_fc1d
        
        out_fc2d = self.fc_2d(outfc1d)
        outfc2d = out_fc2d
        
        out_fc3d = self.fc_3d(outfc2d)
        outfc3d = out_fc3d*b
        out_fc4d = self.fc_4d(outfc3d)        
        out_d = self.fc_5d(out_fc4d)
        out_d = torch.reshape((out_d),(1,1))
        
        out = torch.cat((out_fc4x,out_fc4y,out_d),dim=0)
        return out

class ppc_theta(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 1
        super(ppc_theta, self).__init__()        
        # Fully connected layers
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim*8),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim*8, self.hidden_dim*32),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim*32, self.hidden_dim*8),Sigmoid())
        self.fc_4x = Sequential(Linear(self.hidden_dim*8, self.hidden_dim),Sigmoid())
        self.fc_5x = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        
    def forward(self,x):
        
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        out_fc4x = self.fc_4x(out_fc3x)
        out = self.fc_5x(out_fc4x)
        
        return out

class ppc_vel(Module):
    def __init__(self):
        self.input_dim = 3
        self.hidden_dim = 4
        self.output_dim = 1
        super(ppc_vel, self).__init__()        
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.output_dim))

        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.hidden_dim, self.output_dim))
        
    def forward(self,x,y):
        # first FC
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        
        out_fc1y = self.fc_1y(y)    
        out_fc2y = self.fc_2y(out_fc1y)
        out_fc3y = self.fc_3y(out_fc2y)

        out = torch.cat((out_fc3x,out_fc3y),dim=0)
        return out

class ppc_dist(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 1
        super(ppc_dist, self).__init__()        
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim*8),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim*8, self.hidden_dim*32),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim*32, self.hidden_dim*8),Sigmoid())
        self.fc_4x = Sequential(Linear(self.hidden_dim*8, self.hidden_dim),Sigmoid())
        self.fc_5x = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        
    def forward(self,x):
        # first FC
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        out_fc4x = self.fc_4x(out_fc3x)
        out = self.fc_5x(out_fc4x)
        
        return out

class ppc_dist_v2(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 1
        super(ppc_dist_v2, self).__init__()        
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_4x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_5x = Sequential(Linear(self.hidden_dim, self.output_dim))
        
    def forward(self,x):
        # first FC
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        out_fc3x = self.fc_3x(out_fc2x)
        out_fc4x = self.fc_4x(out_fc3x)
        out = self.fc_5x(out_fc4x)
        
        return out

class ctrlr_ppc_stim(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 1
        super(ctrlr_ppc_stim, self).__init__()        
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.output_dim))

        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.hidden_dim, self.output_dim))
        
    def forward(self,x,y):
        # first FC
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
        outfc2x = out_fc2x*0.45
        out_fc3x = self.fc_3x(outfc2x)
        
        out_fc1y = self.fc_1y(y)  
        out_fc2y = self.fc_2y(out_fc1y)
        outfc2y = out_fc2y*0.45
        out_fc3y = self.fc_3y(outfc2y)

        out = torch.cat((out_fc3x,out_fc3y),dim=0)
        return out
    
class fwd_model(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 1
        super(fwd_model, self).__init__()        
        self.fc_1x = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
#        self.fc_3x = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3x = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())

        self.fc_1y = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_2y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
#        self.fc_3y = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_3y = Sequential(Linear(self.hidden_dim, self.output_dim))#,Tanh())
        
    def forward(self,x,y):
        # first FC
        out_fc1x = self.fc_1x(x)
        out_fc2x = self.fc_2x(out_fc1x)
#        out_fc3xr = self.fc_3x(out_fc2x)
        out_fc3x = self.fc_3x(out_fc2x)
        
        out_fc1y = self.fc_1y(y)    
        out_fc2y = self.fc_2y(out_fc1y)
#        out_fc3y = self.fc_3y(out_fc2y)
        out_fc3y = self.fc_3y(out_fc2y)

        out = torch.cat((out_fc3x,out_fc3y),dim=0)
        return out
    
class brain(Module):
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 4
        self.output_dim = 4
        super(brain,self).__init__()
        self.fc_1xl = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_1yl = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_1xr = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        self.fc_1yr = Sequential(Linear(self.input_dim, self.hidden_dim),Sigmoid())
        
        self.fc_2xl = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_2yl = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_2xr = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        self.fc_2yr = Sequential(Linear(self.hidden_dim, self.hidden_dim),Sigmoid())
        
        self.fc_3xl = Sequential(Linear(self.hidden_dim, self.output_dim))
        self.fc_3yl = Sequential(Linear(self.hidden_dim, self.output_dim))
        self.fc_3xr = Sequential(Linear(self.hidden_dim, self.output_dim))
        self.fc_3yr = Sequential(Linear(self.hidden_dim, self.output_dim))
    
    def forward(self,xr,yr,xl,yl):
        out_fc1xl = self.fc_1xl(xl)
        out_fc1yl = self.fc_1yl(yl)
        out_fc1xr = self.fc_1xr(xr)
        out_fc1yr = self.fc_1yr(yr)
        
        out_fc2xl = self.fc_2xl(out_fc1xl)
        out_fc2yl = self.fc_2yl(out_fc1yl)
        out_fc2xr = self.fc_2xr(out_fc1xr)
        out_fc2yr = self.fc_2yr(out_fc1yr)
        
        out_fc3xl = self.fc_3xl(out_fc2xl)
        out_fc3yl = self.fc_3yl(out_fc2yl)
        out_fc3xr = self.fc_3xr(out_fc2xr)
        out_fc3yr = self.fc_3yr(out_fc2yr)
        
        out = torch.cat((out_fc3xl,out_fc3yl,out_fc3xr,out_fc3yr),dim=0)
        return out
    
    
