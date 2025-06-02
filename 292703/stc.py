def create_part1(name: str):
    with open(name, 'r') as f:
        lines = f.readlines()

    data = [line.strip().split(',') for line in lines]

    header = data[0]
    rows = data[1:]

    sumecw = {
        'Lightweight-Europe': 0,
        'Middleweight-Europe': 0,
        'Heavyweight-Europe': 0,
        'Lightweight-Asia': 0,
        'Middleweight-Asia': 0,
        'Heavyweight-Asia': 0,
        'Lightweight-Americas': 0,
        'Middleweight-Americas': 0,
        'Heavyweight-Americas': 0,
        'Lightweight-Africa': 0,
        'Middleweight-Africa': 0,
        'Heavyweight-Africa': 0,
    }
    countcw = {
        'Lightweight-Europe': 0,
        'Middleweight-Europe': 0,
        'Heavyweight-Europe': 0,
        'Lightweight-Asia': 0,
        'Middleweight-Asia': 0,
        'Heavyweight-Asia': 0,
        'Lightweight-Americas': 0,
        'Middleweight-Americas': 0,
        'Heavyweight-Americas': 0,
        'Lightweight-Africa': 0,
        'Middleweight-Africa': 0,
        'Heavyweight-Africa': 0,
    }
    for r in rows:
        key = r[1] + '-' + r[-2]
        sumecw[key] += float(r[3])/float(r[2])
        countcw[key] += 1

    results = []
    for key in sumecw:
        w = key.split('-')[0]
        c = key.split('-')[1]
        me = sumecw[key]/countcw[key]
        s = str(me)
        results.append([c, w, s])

    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    
    with open('part_one.csv', 'w') as f:
        line = ['continent', 'weight_class', 'mean_efficiency']
        f.write(','.join(line) + '\n') 
        for row in sorted_results:
            line = ','.join(row)
            f.write(line + '\n') 


# def create_part2(name: str):
#     with open(name, 'r') as f:
#         lines = f.readlines()

#     data = [line.strip().split(',') for line in lines]

#     header = data[0]
#     rows = data[1:]

#     T_w_h = 0
#     T_w_m = 0
#     T_w_l = 0

#     datah = []
#     datam = []
#     datal = []

#     for r in rows:
#         if r[1] == 'Heavyweight':
#             datah.append(float(r[2]))
#             if float(r[2]) > T_w_h:
#                 T_w_h = float(r[2])
#         if r[1] == 'Middleweight':
#             datam.append(float(r[2]))
#             if float(r[2]) > T_w_m:
#                 T_w_m = float(r[2])
#         if r[1] == 'Lightweight':
#             datal.append(float(r[2]))
#             if float(r[2]) > T_w_l:
#                 T_w_l = float(r[2])

#     leh = newton_raphson_trunc_exp(datah, T_w_h)
#     lem = newton_raphson_trunc_exp(datam, T_w_m)
#     lel = newton_raphson_trunc_exp(datal, T_w_l)

#     aic_h = 2 - 2*log_likelihood(leh, datah, T_w_h)
#     aic_m = 2 - 2*log_likelihood(lem, datam, T_w_m)
#     aic_l = 2 - 2*log_likelihood(lel, datal, T_w_l)

#     with open('part_two.csv', 'w') as f:
#         line = ['wight_class', 'lambda_hat', 'aic']
#         f.write(','.join(line) + '\n') 
#         line = ['Heavyweight', leh, aic_h]
#         f.write(','.join(line) + '\n') 
#         line = ['Middleweight', lem, aic_m]
#         f.write(','.join(line) + '\n')
#         line = ['Lightweight', lel, aic_l]
#         f.write(','.join(line) + '\n')

# if __name__ == '__main__':
#     create_part1('292703/wrestling_stats.csv')
#     create_part2('292703/wrestling_stats.csv')


