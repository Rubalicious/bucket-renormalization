from tools import *
from mean_field import MeanField
from belief_propagation import BeliefPropagation
import copy

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

def recreate_pandemy(case = 'seattle', alg = 'BP', init_inf = [0], H_a = 0.1, MU = 0.0005, ibound=10):
    # extract data or a subset of the data
    J = extract_data_top20(case, MU)

    # create a complete graph GM
    model = generate_graphical_model(case, J, init_inf, H_a, condition_on_init_inf=False)

    # condition on the initial seed 'V0'
    init_inf = [ith_object_name('V',var) for var in init_inf]
    conditioned_on_init = condition_on_seeds_from(model, init_inf, in_place=False)

    logZ = BucketElimination(conditioned_on_init).run()
    print("logZ computed = {}".format(logZ))

    remaining = [var for var in model.variables if var not in init_inf]
    # print(remaining)
    ratios = []
    filename = 'ratios.csv'
    for var in remaining:
        seeds = copy.copy(init_inf)
        seeds.append(var)

        conditioned_on_init_and_var = condition_on_seeds_from(model, seeds, in_place=False)
        logZi = BucketElimination(conditioned_on_init_and_var).run()

        H_val = conditioned_on_init.get_factor(var.replace('V','B')).log_values[0]
        marg_prob = np.exp(H_val)*np.exp(logZi)/np.exp(logZ)
        print(marg_prob)
        utils.append_to_csv(filename, [var, marg_prob])


def plot_PM_vs_ratios():
    pm_data = utils.read_csv('PM.csv')
    PM = [p[1] for p in pm_data][1:]
    ratio_data = utils.read_csv('ratios.csv')
    ratios = [float(p[1]) for p in ratio_data]
    print("sum = {}".format(np.sum(ratios)))
    ratio1_data = utils.read_csv('ratios1.csv')
    ratios1 = [p[1] for p in ratio1_data]
    plt.plot(range(len(PM)), PM,'*-', label='logZi/logZ')
    plt.plot(range(len(ratios)), ratios,'--', label='e^{-h_i}Zi/Z')
    # plt.plot(range(len(ratios1)), ratios1,'o-', label='recreated')
    plt.legend()
    plt.show()

def marg_prob_formula(H_a, MU):
    # extract data or a subset of the data
    J = extract_data_top20(case, MU)

    # create a complete graph GM
    model = generate_graphical_model(case, J, H_a)

def implement(case = 'seattle', alg = 'BP', init_inf = [0], H_a = 0.1, MU = 0.002, ibound=10):
    '''
        This runs BP or MF
    '''
    # extract data or a subset of the data
    J = extract_data_top20(case, MU)

    # create a complete graph GM
    model = generate_graphical_model(case, J, H_a)

    # choose algorithm
    if alg == 'BP':
        # condition_on_seeds_from(model, init_inf)
        init_inf = [ith_object_name('V',var) for var in init_inf]
        conditioned_on_init = condition_on_seeds_from(model, init_inf)
        logZ = BeliefPropagation(conditioned_on_init).run()
    elif alg == 'MF':
        # condition_on_seeds_from(model, init_inf)
        init_inf = [ith_object_name('V',var) for var in init_inf]
        conditioned_on_init = condition_on_seeds_from(model, init_inf)
        logZ = MeanField(conditioned_on_init).run()
    elif alg == 'GBR':
        # approximate
        compute_marginals(case, model, (init_inf, H_a, MU, ibound))
        return
    elif alg == 'BE':
        # exact
        compute_marginals_BE(case, model, (init_inf, H_a, MU))
        return
    elif alg == 'BE_true':
        logZ = BucketElimination(model).run()
        return
    else:
        raise("Algorthim not defined")

    N = len(model.variables)
    # write results to file
    filename = "{}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format(alg, case, init_inf, H_a, MU)
    utils.append_to_csv(filename, ['Tract index', 'CALI'])
    for index in range(len(J)):
        if ith_object_name('V',index) not in init_inf:
            CALI = 2*logZ['marginals']['MARGINAL_V{}'.format(index)]-1
            utils.append_to_csv(filename, [index, CALI[1]] )
    utils.append_to_csv(filename, ['whole GM', logZ['logZ']])


# plot_mu_for_BE()
# mu_transition_plots()
mu = 0.0005
MUS = [1e-4,2e-4,4e-4,6e-4]
HS = [1e-2,5e-2,10e-2]

# recreate_pandemy(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = 0.1, MU = mu)
# plot_PM_vs_ratios()
# quit()

def plot_result(H_a, MU):
    for alg in ['GBR_ibound=10','GBR_ibound=20', 'MF', 'BP']:
        data = utils.read_csv("{}_seattle_CALI_init_inf=['V0']_H_a={}_MU={}.csv".format(alg, H_a, MU))[1:-1]
        CALI = [float(row[-1]) for row in data]
        plt.plot(range(len(CALI)), CALI, label=alg.replace('_ibound=',''))
    plt.title(r"$\mu$={}, $H_a$={}".format(MU, H_a), fontsize=12)
    plt.xlabel('node number')
    plt.ylabel('CALI')
    plt.legend()
    plt.savefig("./results/MU={}_H_a={}.png".format(MU, H_a))
    plt.show()

def compare_subplots():
    HS = [1e-2,5e-2,1e-1]
    MUS = [1e-4,2e-4,4e-4,6e-4]

    fig, axes = plt.subplots(4, 3, sharex=True, sharey=True)
    # files = os.listdir('./results')
    for alg in ['GBR_ibound=20','GBR_ibound=5','GBR_ibound=10', 'MF', 'BP']:
        for i in range(len(HS)):
            for j in range(len(MUS)):
                data = utils.read_csv("{}_seattle_CALI_init_inf=['V0']_H_a={}_MU={}.csv".format(alg, HS[i], MUS[j]))[1:-1]
                CALI = [float(row[-1]) for row in data]
                # print(CALI)
                plt.subplot(4,3,i+j*len(HS)+1)
                # if alg == 'GBR_ibound=20': alg = 'Exact'
                plt.plot(range(len(CALI)), CALI, label=alg.replace('_ibound=',''))
                plt.title(r"$\mu$={}, $H_a$={}".format(MUS[j], HS[i]), fontsize=8)

                # quit()
    plt.legend(loc='upper right')
    fig.text(0.5, 0.04, r"$H_a$", ha='center')
    fig.text(0.04, 0.5, r"$\mu$", va='center', rotation='vertical')
    plt.show()


def generate_data_for(H_a, MU):
    # implement(case = 'seattle', alg = 'BP', init_inf = [0], H_a = H_a, MU = MU)
    # implement(case = 'seattle', alg = 'MF', init_inf = [0], H_a = H_a, MU = MU)
    implement(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = H_a, MU = MU, ibound = 5)
    # implement(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = H_a, MU = MU, ibound = 10)
    # implement(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = H_a, MU = MU, ibound = 20)


# H_a = 0.01
# MU = 3e-4
# generate_data_for(H_a, MU)
# plot_result(H_a, MU)
# implement(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = H_a, MU = MU, ibound = 20)


# basic checks
# fix mu = 0, vary H_a
# fix H, increase MU, ==> CALI --> +1
#
# HS = [1e-2,5e-2,1e-1]
# MUS = [1e-4,2e-4,4e-4,6e-4]
# for mu in MUS:
#     for H_a in HS:
#         print("running experiments for mu={}, H_a = {}".format(mu, H_a))
#         generate_data_for(H_a, mu)
        # plot_result(H_a, mu)

# compare_subplots()


# mu=0.0001, h=0.01
# 4
# p_exact = [0.4971938023030517, 0.5025450897547602, 0.4993989150688658, 0.
# ,!4984037249367082, 0.5030564925069722, 0.5012683029976511, 0.4993210184738254,
# ,!0.4969575916085563, 0.5027389915401242, 0.4972146734362853, 0.
# ,!4969562949348514, 0.49730486459999046, 0.507188944634411, 0.5021764934977803,
# ,!0.49507478575480096, 0.4980818731509484, 0.5035539920237058, 0.
# ,!496722135573331, 0.49830680651655873]
# p_mf_inferlo = [0.4971848843668987, 0.5025803985517158, 0.49940475901768644, 0.
# ,!49840566441649214, 0.5030993262848655, 0.5012860090782918, 0.
# ,!49932613469979853, 0.4969567074385086, 0.5027569246927565, 0.4972113748617529,
# ,!0.49693348742738597, 0.4973025127037605, 0.5073701759049752, 0.
# ,!5022011570559047, 0.4950653708570877, 0.4980810355525359, 0.5035730293031024,
# ,!0.4967137769593073, 0.4983062866307225]
# p_gbr5 = [0.49755670395797413, 0.5033820572532484, 0.4996607251356631, 0.
# ,!4985286775090472, 0.5032581003876697, 0.5015222618518154, 0.49952388794138497,
# ,!0.4970018475808352, 0.5031981643526047, 0.4972167403656993, 0.
# ,!4976062334867557, 0.4976388327942517, 0.5085828828876346, 0.5026645364820154,
# ,!0.4953179537162413, 0.4980934259135538, 0.5037475029877392, 0.
# ,!4969942353867113, 0.4983691695947848]
# p_gbr10 = [0.49754123620949964, 0.5027672605119998, 0.499595373913597, 0.
# ,!49859796694371183, 0.5031447577525384, 0.5013972128111671, 0.
# ,!49973979047790124, 0.49708242023006094, 0.5030035075410692, 0.
# ,!49741105068906655, 0.49726375209177587, 0.4974573163070583, 0.
# ,!5077479061306042, 0.5024778346742739, 0.4954065221213778, 0.4983903351980553,
# ,!0.5038460865298673, 0.49680115758006815, 0.49832495336210963]
# p_bp_sungsoo = [0.4974514151110049, 0.5028982537937949, 0.499531288950727, 0.
# ,!498553455122775, 0.5033903336490293, 0.5015309527006923, 0.49956817937121345,
# ,!0.4970465647909179, 0.5029733808749947, 0.49734949866652, 0.4973633789590499,
# ,!0.4974817324055931, 0.5080633133243598, 0.5025822906243342, 0.
# ,!49521616588968703, 0.49817964599862236, 0.5037595017283863, 0.
# ,!49689979826740416, 0.498338953525337]

# plot_result(H_a = 5e-2, MU = 6e-4)
# plot_result(H_a = 1e-1, MU = 6e-4)

# compare(case = 'seattle', init_inf = [0], H_a = H_a , MU = mu)
# compare(case = 'seattle', init_inf = [0], H_a = 0.1 , MU = mu)
# compare_to_misha_results()

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



'''
def compare(case = 'seattle', init_inf = [0], H_a = 0.1 , MU = 0.0005):
    alg1 = 'BP'
    alg2 = 'MF'
    alg3 = 'GBR'
    alg4 = 'GBR'
    alg5 = 'recreated_misha'

    init_inf = [ith_object_name('V', var) for var in init_inf]
    filename = "{}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format(alg1, case, init_inf, H_a, MU)
    data1 = utils.read_csv(filename)
    filename = "{}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format(alg2, case, init_inf, H_a, MU)
    data2 = utils.read_csv(filename)
    filename = "{}_ibound={}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format(alg3,5, case, init_inf, H_a, MU)
    data3 = utils.read_csv(filename)
    filename = "{}_ibound={}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format(alg3,10, case, init_inf, H_a, MU)
    data4 = utils.read_csv(filename)
    # filename = "recreated_misha.csv"
    # data5 = utils.read_csv(filename)

    N = len(data3)-1
    error = []
    BP_p = []
    MF_p = []
    GBR5_p = []
    GBR10_p = []
    # PM_p = []
    for line in range(1,N):
        # print(float(data1[1:-1][line][1].split(" ")[1]))
        # quit()
        marg1 = [float(data1[1:-1][line-1][1].split(" ")[1])]
        marg2 = [float(data2[1:-1][line-1][1].split(" ")[1])]
        # marg2 = [float(val) for val in data2[line][1][1:-1].split(" ")]
        marg3 = [float(data3[line][-1])]
        marg4 = [float(data4[line][-1])]
        marg5 = [float(data5[line-1][-1])]

        BP_p.append(marg1[0])
        MF_p.append(marg2[0])
        GBR5_p.append(marg3[0])
        GBR10_p.append(marg4[0])
        # PM_p.append(2*marg5[0]-1)


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
    misha = [2*p[0]-1 for p in truth['marg_prob']]

    start_node = int(0)
    plt.plot(range(len(BP_p)), BP_p, label='BP')
    plt.plot(range(len(MF_p)), MF_p, label='MF')
    plt.plot(range(len(GBR5_p)), GBR5_p, label='GBR5')
    plt.plot(range(len(GBR10_p)), GBR10_p, label='GBR10')
    # plt.plot(range(len(PM_p)), PM_p, label='PM')
    # plt.plot(range(len(misha)), misha, label='bruteforce')
    plt.xlabel('node number')
    plt.ylabel('probability')
    plt.legend()
    plt.title("Comparing CALI of algorithms\n for init_inf = {}, " r"$H_a$ = {}, $\mu$ = {}".format(init_inf, H_a, MU))
    plt.savefig('./results/comparison.png')
    plt.show()

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
    misha = [p[0] for p in truth['marg_prob']]
    plt.plot(range(len(probs)), probs, label='ruby output')
    plt.plot(range(len(misha)), misha, label='misha output')
    plt.legend()
    plt.show()

'''
