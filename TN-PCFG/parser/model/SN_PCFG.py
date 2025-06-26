import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from parser.pcfgs.simple_pcfg import SimplePCFG_Triton


class Simple_N_PCFG(nn.Module):
    def __init__(self, args, dataset):
        super(Simple_N_PCFG, self).__init__()
        self.pcfg = SimplePCFG_Triton()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab) if hasattr(dataset, 'word_vocab') else len(dataset.V)
        self.s_dim = args.s_dim
        rule_dim = self.s_dim
        
        ## root
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        #terms
        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
        )

        self.vocab_emb =  nn.Parameter(torch.randn(self.s_dim, self.V))

        self.rule_state_emb = nn.Parameter(torch.randn(self.NT+self.T, self.s_dim))
        
        self.left_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU()) 
        self.right_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU())
        self.parent_mlp1 =  nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                          nn.ReLU(),
                                      )

        self._initialize()  


    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input, **kwargs):

        x = input['word']
        b, n = x.shape[:2]


        def roots():
            roots = (self.root_emb  @ self.rule_state_emb[:self.NT].t())
            roots = roots.log_softmax(-1)
            return roots.expand(b, roots.shape[-1])


        def terms():
            term_emb = self.rule_state_emb[self.NT:]
            term_prob = ((self.term_mlp(term_emb) + term_emb) @ self.vocab_emb).log_softmax(-1)
            return term_prob[torch.arange(self.T)[None,None], x[:, :, None]]

        def rules():
            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb[:self.NT]
            parent1 = self.parent_mlp1(nonterm_emb) + nonterm_emb   


            # parent2 = self.parent_mlp2(nonterm_emb) + nonterm_emb   
            left = (self.left_mlp(rule_state_emb) + rule_state_emb) @  parent1.t()
            right = (self.right_mlp(rule_state_emb) + rule_state_emb) @ parent1.t()
            # right = left

            # head = head.softmax(-1)
            left = left.softmax(-2)
            right = right.softmax(-2)

            left_m =  left[:self.NT, :]
            left_p =  left[self.NT:, :]
            
            right_m = right[:self.NT, :]
            right_p = right[self.NT:, :]
            
            return (left_m, left_p, right_m, right_p)

        root, unary, (left_m, left_p, right_m, right_p) = roots(), terms(), rules()

        return {'unary': unary,
                'root': root,
                'left_m': left_m,
                'right_m': right_m,
                'left_p': left_p,
                'right_p' : right_p,
                'kl': 0}
    
    def generate(self):
        # Pretty much get a forward pass
        roots = (self.root_emb  @ self.rule_state_emb[:self.NT].t())
        roots = roots.softmax(-1)

        term_emb = self.rule_state_emb[self.NT:]
        term_prob = ((self.term_mlp(term_emb) + term_emb) @ self.vocab_emb) #prior to softmax must make the unk token logit to be -inf
        term_prob[:, 1] = float('-inf')
        term_prob = term_prob.softmax(-1)

        rule_state_emb = self.rule_state_emb
        nonterm_emb = rule_state_emb[:self.NT]
        parent1 = self.parent_mlp1(nonterm_emb) + nonterm_emb   
        left = (self.left_mlp(rule_state_emb) + rule_state_emb) @  parent1.t()
        right = (self.right_mlp(rule_state_emb) + rule_state_emb) @ parent1.t()
        left = left.softmax(-2)
        right = right.softmax(-2)

        first = torch.multinomial(roots, 1).item() # gets the first non-terminal
        string = [first]
        tree_string_list = [first]
        while any(s < self.NT for s in string) and len(string) <= 150:  # Continue until all symbols are terminals or max len
            new_string = []
            find_index = 0
            for s in string:
                if s < self.NT:
                    # s is a non-terminal, expand it
                    left_child = torch.multinomial(left[:, s], 1).item()
                    right_child = torch.multinomial(right[:, s], 1).item()

                    new_string.append(left_child)
                    new_string.append(right_child)
                    
                    curr_index = tree_string_list.index(s, find_index)
                    insert_list = [-1, left_child, right_child, -2]
                    tree_string_list = tree_string_list[:curr_index] + insert_list + tree_string_list[curr_index+1:]
                    find_index = curr_index + 4
                    
                else:
                    # s is a terminal, add it to the new string
                    new_string.append(s)

            string = new_string
            if len(string) > 150:
                return None    
        
        # Convert terminal indices to words and create string tree representation
        words = [] 
        for s in string:
            # since s is a terminal, we subtract NT to get the correct index since term_prob is indexed from 0 and only contains terminals
            word = torch.multinomial(term_prob[s - self.NT], 1).item() 
            words.append(word)
            tree_string_list[tree_string_list.index(s)] = word
                    
        return words, tree_string_list

    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg._inside(rules=rules, lens=input['seq_len'])
        logZ =  -result['partition'].mean()
        return logZ


    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input)
        if decode_type == 'viterbi':
            assert NotImplementedError

        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError