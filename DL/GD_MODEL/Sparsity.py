import torch
class Sparse_net(torch.nn.Module):
    '''
    Finding the sparse representation based on dictionary in each step as input in forward function
    ---------------
    Parameter:    
        number_of_atoms : number of atoms that you need for Dictionary
        num_patches : assigned to the number of patches to construct the Sparse representation matrix    
    '''
    def __init__(self, number_of_atoms, num_patches):
        super(Sparse_net, self).__init__()
        
        self.number_of_atoms = number_of_atoms
        self.sv = torch.rand(self.number_of_atoms, num_patches)
        
    def forward(self, x):
        return x.cuda().to(torch.float32).T @ self.sv.cuda().to(torch.float32) 



            
            
    
        