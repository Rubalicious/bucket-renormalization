from tools import *
from mean_field import MeanField
from belief_propagation import BeliefPropagation

def test0():
    nb_vars = 70
    delta = 1.0

    for i in range(2,20):
        print(10*i)
        toy = generate_complete(10*i, delta)
        # toy = generate_star(10*i)
        t1 = time.time()
        Z = BucketRenormalization(toy, ibound=10).run(max_iter=1)
        t2 = time.time()

        print(Z)
        print(t2-t1)
    quit()

def testing_run_time():
    init_inf = [0]
    runtime = []
    TAUS = [60, 70, 80, 90, 100, 110, 120]
    for H_a in [1.0]:
        for MU in [0.00138985]:
            for TAU in TAUS:
                G = extract_seattle_data(TAU, MU)
                seattle = generate_seattle(G, init_inf, H_a)
                # =====================================
                # Compute partition function for Seattle GM
                # =====================================
                t1 = time.time()
                Z = BucketRenormalization(seattle, ibound=10).run(max_iter=1)
                t2 = time.time()
                # =====================================
                print('partition function = {}'.format(Z))
                print('time taken for GBR = {}'.format(t2-t1))
                runtime.append(t2-t1)
    plt.plot(TAUS, runtime)
    plt.ylabel("GBR Runtime")
    plt.xlabel(R"minium threshold $\tau$")
    plt.show()
    quit()

def testing_partition_function_dependence_on_TAU():
    init_inf = [0]
    pf = []
    TAUS = [70, 80, 90, 100, 110, 120]
    for H_a in [1.0]:
        for MU in [0.00138985]:
            for TAU in TAUS:
                G = extract_seattle_data(TAU, MU)
                seattle = generate_seattle(G, init_inf, H_a)
                # =====================================
                # Compute partition function for Seattle GM
                # =====================================
                t1 = time.time()
                Z = BucketRenormalization(seattle, ibound=10).run(max_iter=1)
                t2 = time.time()
                # =====================================
                print('partition function = {}'.format(Z))
                print('time taken for GBR = {}'.format(t2-t1))
                pf.append(Z)
    plt.plot(TAUS, pf)
    plt.ylabel("Z Value")
    plt.xlabel(R"minium threshold $\tau$")
    pls.savefig("Z_dependence_on_tau.png")
    plt.show()
    quit()

def implement(case = 'seattle', alg = 'BP', init_inf = [0], H_a = 0.1, MU = 0.002, ibound=10, subcase = 'top20'):
    '''
        This runs BP or MF
    '''
    # extract data or a subset of the data
    J = extract_data_top20(case, MU)

    # create a complete graph GM
    model = generate_graphical_model(case, J, init_inf, H_a, condition_on_init_inf=True)

    # choose algorithm
    if alg == 'BP':
        Z = BeliefPropagation(model).run()
    elif alg == 'MF':
        Z = MeanField(model).run()
    elif alg == 'GBR':
        # approximate
        compute_marginals(case, model, (init_inf, H_a, MU, ibound))
        return
    elif alg == 'BE':
        # exact
        compute_marginals_BE(case, model, (init_inf, H_a, MU))
        return
    elif alg == 'BE_true':
        Z = BucketElimination(model).run()
        print(Z)
        return
    else:
        raise("Algorthim not defined")

    N = len(model.variables)
    # write results to file
    filename = "{}_{}_marg_prob_init_inf={}_H_a={}_MU={}.csv".format(alg, case, init_inf, H_a, MU)
    utils.append_to_csv(filename, ['Tract index', 'P'])
    for index in range(len(J)):
        if index not in init_inf:
            # print('P( x_{} = {} ) = {}'.format(index, 1, ))
            utils.append_to_csv(filename, [index, Z['marginals']['MARGINAL_V{}'.format(index)]] )
    utils.append_to_csv(filename, ['whole GM', Z['logZ']])


def compare(case = 'seattle', init_inf = [81], H_a = 0.1 , MU = 0.002):
    alg1 = 'BP'
    alg2 = 'MF'
    alg3 = 'GBR'
    alg4 = 'BE'

    filename = "{}_{}_marg_prob_init_inf={}_H_a={}_MU={}.csv".format(alg1, case, init_inf, H_a, MU)
    data1 = utils.read_csv(filename)
    filename = "{}_{}_marg_prob_init_inf={}_H_a={}_MU={}.csv".format(alg2, case, init_inf, H_a, MU)
    data2 = utils.read_csv(filename)
    filename = "{}_ibound={}_{}_marg_prob_init_inf={}_H_a={}_MU={}.csv".format(alg3,10, case, init_inf, H_a, MU)
    data3 = utils.read_csv(filename)
    filename = "{}_{}_marg_prob_init_inf={}_H_a={}_MU={}.csv".format(alg4, case, init_inf, H_a, MU)
    data4 = utils.read_csv(filename)

    N = len(data1)-1
    error = []
    BP_p = []
    MF_p = []
    GBR_p = []
    BE_p = []
    for line in range(1,N):
        marg1 = [float(val) for val in data1[line][1][1:-1].split(" ")]
        marg2 = [float(val) for val in data2[line][1][1:-1].split(" ")]
        marg3 = [float(data3[line][-1])]
        marg4 = [float(data4[line][-1])]

        BP_p.append(marg1[0])
        MF_p.append(marg2[0])
        GBR_p.append(marg3[0])
        BE_p.append(marg4[0])

    start_node = int(data1[1][0])
    plt.plot(range(len(BP_p)), BP_p, label='BP')
    plt.plot(range(len(MF_p)), MF_p, label='MF')
    plt.plot(range(len(GBR_p)), GBR_p, label='GBR')
    plt.plot(range(len(BE_p)), BE_p, label='BE')
    plt.xlabel('node number')
    plt.ylabel('probability')
    plt.legend()
    plt.title('Comparing Marginals of algorithms\n for init_inf = {}, H_a = {}, MU = {}'.format(init_inf, H_a, MU))
    # plt.legend({'BP', 'MF', 'GBR','BE'})
    plt.savefig('./results/comparison.png')
    plt.show()

    # plt.plot(range(start_node, start_node+len(BP_p)), np.log(BP_p))
    # plt.plot(range(start_node, start_node+len(MF_p)), np.log(MF_p))
    # plt.plot(range(start_node, start_node+len(GBR_p)), np.log(GBR_p))
    # plt.plot(range(start_node, start_node+len(BE_p)), np.log(BE_p))
    # plt.xlabel('node number')
    # plt.ylabel('log of probability')
    # plt.title('Comparing Marginals of algorithms\n for init_inf = {}, H_a = {}, MU = {}'.format(init_inf, H_a, MU))
    # plt.legend({'BP', 'MF', 'GBR','BE'})
    # plt.show()

def ibound_runtime_plots():
    runtimes = []
    ibounds = []
    for filename in os.listdir('./results_ibound'):
        ibound = int(filename.split('=')[1].split('_')[0])
        ibounds.append( ibound )

        data = utils.read_csv(filename, dir_name='./results_ibound')
        average_runtime = np.mean([float(row[2]) for row in data[1:]])
        runtimes.append(average_runtime)

    ib_sorted = [ ib for ib,rt in sorted(zip(ibounds,runtimes))]
    rt_sorted = [ rt for ib,rt in sorted(zip(ibounds,runtimes))]
    print(ib_sorted)
    print(rt_sorted)

    plt.plot(ib_sorted, rt_sorted)
    plt.xlabel('ibound parameter')
    plt.ylabel('average runtime')
    plt.title('average runtime of GBR')
    plt.show()

def mu_transition_plots():
    probs_plus = []
    probs_minus = []

    MUS = np.linspace(0.0001, 0.001, 20)[:9]
    for MU in MUS:
        filename = "{}_{}_marg_prob_init_inf={}_H_a={}_MU={}.csv".format('BP', 'seattle', [81], 0.1, MU)
        data = utils.read_csv(filename, dir_name='./results_transition')
        probs_in_data = [float(val) for val in data[1:][1][1][1:-1].split(" ")]
        print(probs_in_data)
        probs_plus.append(probs_in_data[1])
        probs_minus.append(probs_in_data[0])
    plt.plot(MUS, probs_plus, MUS, probs_minus)
    plt.xlabel(r"$\mu$ value")
    plt.ylabel("Probability")
    plt.title("transition plot")
    plt.legend({'P(+)', 'P(--)'})
    plt.show()

def plot_mu_for_BE():
    H_a = 0.1
    init_inf = [0]
    case = 'seattle'
    alg1 = 'BE'
    for mu in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006]:
        probs = []
        filename = "{}_{}_marg_prob_init_inf={}_H_a={}_MU={}.csv".format(alg1, case, init_inf, H_a, mu)
        data = utils.read_csv(filename)[1:-1]
        # print(data)
        for row in data:
            # print(row[-1])
            probs.append(float(row[-1]))

        plt.plot(range(len(probs)), probs, label=r"$\mu={}$".format(mu))
    plt.legend()
    plt.savefig('range_of_mu.png')
    plt.show()

def compare_to_misha_results():
    truth = {
        'log_pf': 15.699029307675897,
        'marg_prob': np.array([[0.83806142, 0.16193858],
                            [0.86201057, 0.13798943],
                            [0.79396838, 0.20603162],
                            [0.8050639 , 0.1949361 ],
                            [0.8640532 , 0.1359468 ],
                            [0.84843595, 0.15156405],
                            [0.83755868, 0.16244132],
                            [0.75066252, 0.24933748],
                            [0.84276857, 0.15723143],
                            [0.78152325, 0.21847675],
                            [0.87887964, 0.12112036],
                            [0.80459683, 0.19540317],
                            [0.9360527 , 0.0639473 ],
                            [0.88312804, 0.11687196],
                            [0.78007011, 0.21992989],
                            [0.76910475, 0.23089525],
                            [0.8444268 , 0.1555732 ],
                            [0.81671871, 0.18328129],
                            [0.69913106, 0.30086894]])

    }
    myZ = 15.676771561522301
    alg1 = 'BE'
    case = 'seattle'
    init_inf = [0]
    H_a = .1
    MU = 0.0005
    filename = "{}_{}_marg_prob_init_inf={}_H_a={}_MU={}.csv".format(alg1, case, init_inf, H_a, MU)
    data = utils.read_csv(filename)[1:-1]
    probs = []
    for i in range(1,len(data)):
        print(truth['marg_prob'][i][0]/float(data[i][-1]))
        probs.append(float(data[i][-1]))



# plot_mu_for_BE()
# mu_transition_plots()
# compare_to_misha_results()

mu = 0.0005
implement(case = 'seattle', alg = 'BP', init_inf = [0], H_a = 0.1, MU = mu)
implement(case = 'seattle', alg = 'MF', init_inf = [0], H_a = 0.1, MU = mu)
implement(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = 0.1, MU = mu)
implement(case = 'seattle', alg = 'BE', init_inf = [0], H_a = 0.1, MU = mu)
compare(case = 'seattle', init_inf = [0], H_a = 0.1 , MU = mu)

# for mu in np.linspace(0.0001, 0.001, 20):
#     print("mu={}".format(mu))
# for mu in [0.0001, 0.0002, 0.0003, 0.0004, 0.0006]:
#     implement(case = 'seattle', alg = 'BE', init_inf = [0], H_a = 0.1, MU = mu)


# implement(case = 'seattle', alg = 'MF', init_inf = [81], H_a = 0.1, MU = mu)
# implement(case = 'seattle', alg = 'GBR', init_inf = [81], H_a = 0.1, MU = mu, ibound = 10)
# compare(case = 'seattle', init_inf = [81], H_a = 0.1 , MU = mu)
# compare(alg1='BP', alg2='MF', case = 'seattle', init_inf = [81], H_a = 0.1 , MU = mu)
# quit()
# implement(case = 'seattle', alg = 'BP', init_inf = [81], H_a = 0.1, MU = mu)
# implement(case = 'seattle', alg = 'MF', init_inf = [81], H_a = 0.1, MU = mu)
# implement(case = 'seattle', alg = 'GBR', init_inf = [81], H_a = 0.1, MU = mu)
# implement(case = 'seattle', alg = 'BE', init_inf = [81], H_a = 0.1, MU = mu)
# compare(case = 'seattle', init_inf = [81], H_a = 0.1 , MU = mu)
