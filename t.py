ag_t = open('test.ab-ag.ag', 'w')
ab_t = open('test.ab-ag.ab', 'w')
ab_t_w = open('test.ab-ag.ab.weights', 'w')
ag_v = open('valid.ab-ag.ag', 'w')
ab_v = open('valid.ab-ag.ab', 'w')
ab_v_w = open('valid.ab-ag.ab.weights', 'w')
    
for i in range(len(self.db_ids)):
    s = self.get_structure(self.db_ids[i])
    if s['heavy'] is None or s['antigen'] is None:
        continue
    h = s['heavy']
    h_seq = h.FW1_seq + \
            h.H1_seq + \
            h.FW2_seq + \
            h.H2_seq + \
            h.FW3_seq + \
            h.H3_seq + \
            h.FW4_seq + '\n'

    w = '0' * len(h.FW1_seq) + \
        '1' * len(h.H1_seq) + \
        '0' * len(h.FW2_seq) + \
        '1' * len(h.H2_seq) + \
        '0' * len(h.FW3_seq) + \
        '1' * len(h.H3_seq) + \
        '0' * len(h.FW4_seq) + '\n'
            
    antigen = _aa_tensor_to_sequence(s['antigen']['aa'])

    if random.random() < 0.5:
        ab_t.write(h_seq)
        ag_t.write(antigen[:250]+'\n')
        ab_t_w.write(w)
    else:
        ab_v.write(h_seq)
        ag_v.write(antigen[:250]+'\n')
        ab_v_w.write(w)

ag_t.close()
ab_t.close()
ag_v.close()
ab_v.close()
ab_v_w.close()
ab_t_w.close()