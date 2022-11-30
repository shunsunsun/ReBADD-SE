MAX_RING_COUNTS = 20


def parse_key(x):
    if x[0] == '%':
        return int(x[1:])
    else:
        return int(x)
    

def reduce_ring_number(smiles):
    ## create a generator
    chars_iter = (c for c in smiles)
    
    ## init output
    chars = []
    
    ## init
    mapping = {}
    rids_not_used = [f'{i}' for i in range(1,10)] + [f'%{i}' for i in range(10,MAX_RING_COUNTS)]
    
    ## get an initial character
    c = next(chars_iter, None)
    
    ## do
    while c is not None:
        ## exception of %
        if c == '%':
            c += next(chars_iter) + next(chars_iter) 
        ## Case1: ion symbols
        if c == '[':
            while c != ']':
                chars.append(c)
                c = next(chars_iter)
            assert c == ']'
            chars.append(c)
        ## Case2: ring
        elif c.isdigit() or ('%' in c):
            ## close with a new ring number
            if c in mapping:
                c_new = mapping.pop(c)
                rids_not_used.append(c_new)
                rids_not_used = list(sorted(rids_not_used, key=parse_key))
            ## open with a new ring number
            else:
                c_new = rids_not_used.pop(0)
                mapping[c] = c_new
            chars.append(c_new)
        ## Case3: other symbols
        else:
            chars.append(c)
        ## next character
        c = next(chars_iter, None)
        
    ## error check
    assert len(mapping) == 0
    
    return ''.join(chars)
    
    
def find_all_not_used_rids(smiles):
    ## create a generator
    chars_iter = (c for c in smiles)
    
    ## init output
    rids_not_used = [f'{i}' for i in range(1,10)] + [f'%{i}' for i in range(10,MAX_RING_COUNTS)]
    
    ## get an initial character
    c = next(chars_iter, None)
    
    ## search all not used numbers
    while c is not None:
        ## exception of %
        if c == '%':
            c += next(chars_iter) + next(chars_iter) 
        ## Case1: ion symbols
        if c == '[':
            while c != ']':
                c = next(chars_iter)
            assert c == ']'
        ## Case2: ring
        elif c in rids_not_used:
            _ = rids_not_used.pop(rids_not_used.index(c))
        ## next character
        c = next(chars_iter, None)
    
    return rids_not_used


def rearrange_ring_number(smiles):
    ## create a generator
    chars_iter = (c for c in smiles)
    
    ## init output
    chars = []
    
    ## init
    mapping = {}
    rids = [f'{i}' for i in range(1,10)] + [f'%{i}' for i in range(10,MAX_RING_COUNTS)]
    rid2stat = {idx:0 for idx in rids} # 0: not_used, 1: open, 2: closed
    
    ## search all not used numbers
    rids_not_used = find_all_not_used_rids(smiles)
    
    ## get an initial character
    c = next(chars_iter, None)
    
    ## do arrangement
    while c is not None:
        ## exception of %
        if c == '%':
            c += next(chars_iter) + next(chars_iter) 
        ## Case1: ion symbols
        if c == '[':
            while c != ']':
                chars.append(c)
                c = next(chars_iter)
            assert c == ']'
            chars.append(c)
        ## Case2: other symbols
        else:
            status = rid2stat.get(c, -1)
            ## ring open
            if status == 0:
                rid2stat[c] = 1
            ## ring close
            elif status == 1:
                rid2stat[c] = 2
            ## ring id rearrangement
            elif status == 2:
                ## close
                if c in mapping:
                    c_new = mapping.pop(c)
                    assert rid2stat[c_new] == 1
                    rid2stat[c_new] = 2
                ## open
                else:
                    c_new = rids_not_used.pop(0)
                    mapping[c] = c_new
                    assert rid2stat[c_new] == 0
                    rid2stat[c_new] = 1
                c = c_new
            chars.append(c)
        ## next character
        c = next(chars_iter, None)          
    
    ## error check
    assert len(mapping) == 0
    
    return ''.join(chars)


def change_tetrahedral_carbon(smiles):
    smiles = smiles.replace('[C@@H]', 'tmp').replace('[C@H]', 'tmp2').replace('[CH]', 'tmp3')
    smiles = smiles.replace('tmp', '[C@H]').replace('tmp2', '[CH]').replace('tmp3', '[C@@H]')
    
    smiles = smiles.replace('[C@@]', 'tmp').replace('[C@]', 'tmp2').replace('[C]', 'tmp3')
    smiles = smiles.replace('tmp', '[C@]').replace('tmp2', '[C]').replace('tmp3', '[C@@]')
    return smiles


def branch_based_standardization(smiles, use_rearrange_ring_number=True, use_reduce_ring_number=True, verbose=0):
    ## Rearrange ring numbers in smiles
    if use_rearrange_ring_number:
        smiles = rearrange_ring_number(smiles)
        
    ## Decompose the given SMILES string into 3 substrings (source, branch1, branch2)
    substrings = ['', '', '']
    MAX_PNUM = 2 # len(substrings) - 1
    
    snum = 0 # stack height
    pnum = 0 # partition number
    pnum_prev = 0
    ion_flag = False
    
    for i, symbol in enumerate(smiles):
        
        if verbose > 0:
            print(i, symbol, pnum, snum, MAX_PNUM)
        
        if symbol == '[':
            assert not ion_flag
            ion_flag = True
            substrings[pnum] += symbol
        elif symbol == ']':
            assert ion_flag
            ion_flag = False
            substrings[pnum] += symbol
        elif ion_flag:
            substrings[pnum] += symbol
        else:
            if symbol == '(':
                snum += 1
                if snum == 1:
                    if smiles[i-1] == ')' and pnum == 2 and pnum_prev == 1:
                        pnum_prev = pnum
                        pnum = min(pnum+1, MAX_PNUM)
                        substrings.append('')
                        MAX_PNUM = 3
                    else:
                        pnum_prev = pnum
                        pnum = min(pnum+1, MAX_PNUM)
                substrings[pnum] += symbol
            elif symbol == ')':
                substrings[pnum] += symbol
                snum -= 1
                if snum == 0:
                    pnum_prev = pnum
                    pnum = min(pnum+1, MAX_PNUM)
            else:
                substrings[pnum] += symbol            
            
    ## stack check
    assert snum == 0
    assert not ion_flag
    
    if verbose > 0:
        print(substrings)
    
    ## branch exists?
    if len(substrings[1]) == 0:
        return substrings[0]
    elif len(substrings[2]) == 0:
        return substrings[0] + substrings[1]
    
    if len(substrings) == 3:
        ## remove the outermost bracket
        substrings[1] = substrings[1][1:-1]
        ## sort by length
        if len(substrings[1]) > len(substrings[2]):
            substrings[1], substrings[2] = substrings[2], substrings[1]
            substrings[0] = change_tetrahedral_carbon(substrings[0])
        ## assemble
        reconstructed = substrings[0]
        reconstructed += '(' + branch_based_standardization(substrings[1], False, False, verbose) + ')'
        reconstructed += branch_based_standardization(substrings[2], False, False, verbose)
    elif len(substrings) == 4:
        ## remove the outermost bracket
        substrings[1] = substrings[1][1:-1]
        substrings[2] = substrings[2][1:-1]
        ## sort by length
        if len(substrings[1]) > len(substrings[2]):
            substrings[1], substrings[2] = substrings[2], substrings[1]
            substrings[0] = change_tetrahedral_carbon(substrings[0])
        if len(substrings[2]) > len(substrings[3]):
            substrings[2], substrings[3] = substrings[3], substrings[2]
            substrings[0] = change_tetrahedral_carbon(substrings[0])
            substrings[1] = change_tetrahedral_carbon(substrings[1])
        if len(substrings[1]) > len(substrings[2]):
            substrings[1], substrings[2] = substrings[2], substrings[1]
            substrings[0] = change_tetrahedral_carbon(substrings[0])
        ## assemble
        reconstructed = substrings[0]
        reconstructed += '(' + branch_based_standardization(substrings[1], False, False, verbose) + ')'
        reconstructed += '(' + branch_based_standardization(substrings[2], False, False, verbose) + ')'
        reconstructed += branch_based_standardization(substrings[3], False, False, verbose)
        
    if use_reduce_ring_number:
        reconstructed = reduce_ring_number(reconstructed)
        
    ## rdkit SMILES parse error
    #assert '((' not in reconstructed and '))' not in reconstructed

    return  reconstructed