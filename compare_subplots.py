def compare_subplots():
    HS = [1e-2,5e-2,1e-1]
    MUS = [1e-4,2e-4,4e-4,6e-4]

    fig, axes = plt.subplots(4, 3, sharex=True, sharey=True)
    # files = os.listdir('./results')
    for alg in ['GBR_ibound=5','GBR_ibound=10','GBR_ibound=20', 'MF', 'BP']:
        for i in range(len(HS)):
            for j in range(len(MUS)):
                data = utils.read_csv("{}_seattle_CALI_init_inf=['V0']_H_a={}_MU={}.csv".format(alg, HS[i], MUS[j]))[1:-1]
                CALI = [float(row[-1]) for row in data]
                # print(CALI)
                plt.subplot(4,3,i+j*len(HS)+1)
                # if alg == 'GBR_ibound=20': alg = 'Exact'
                plt.plot(range(len(CALI)), CALI, label=alg.replace('_ibound=',''))
                plt.title(r"$\mu$={}, $H_a$={}".format(MUS[j], HS[i]), fontsize=8)
