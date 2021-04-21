from tools import *
from mean_field import MeanField
from belief_propagation import BeliefPropagation
import copy
import json
import seaborn as sns

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
#
# def recreate_pandemy(case = 'seattle', alg = 'BP', init_inf = [0], H_a = 0.1, MU = 0.0005, ibound=10):
#     # extract data or a subset of the data
#     J = extract_data_top20(case, MU)
#
#
#     # create a complete graph GM
#     model = generate_graphical_model(case, J, init_inf, H_a, condition_on_init_inf=False)
#
#     # condition on the initial seed 'V0'
#     init_inf = [ith_object_name('V',var) for var in init_inf]
#     conditioned_on_init = condition_on_seeds_from(model, init_inf, in_place=False)
#
#     logZ = BucketElimination(conditioned_on_init).run()
#     print("logZ computed = {}".format(logZ))
#
#     remaining = [var for var in model.variables if var not in init_inf]
#     # print(remaining)
#     ratios = []
#     filename = 'ratios.csv'
#     for var in remaining:
#         seeds = copy.copy(init_inf)
#         seeds.append(var)
#
#         conditioned_on_init_and_var = condition_on_seeds_from(model, seeds, in_place=False)
#         logZi = BucketElimination(conditioned_on_init_and_var).run()
#
#         H_val = conditioned_on_init.get_factor(var.replace('V','B')).log_values[0]
#         marg_prob = np.exp(H_val)*np.exp(logZi)/np.exp(logZ)
#         print(marg_prob)
#         utils.append_to_csv(filename, [var, marg_prob])


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

# def marg_prob_formula(H_a, MU):
#     # extract data or a subset of the data
#     J = extract_data_top20(case, MU)
#
#     # create a complete graph GM
#     model = generate_graphical_model(case, J, H_a)


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


def generate_data_for(H_a, MU, init_inf=[0]):
    implement(case = 'seattle', alg = 'BP', init_inf = [0], H_a = H_a, MU = MU)
    # implement(case = 'seattle', alg = 'MF', init_inf = [0], H_a = H_a, MU = MU)
    # implement(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = H_a, MU = MU, ibound = 5)
    # implement(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = H_a, MU = MU, ibound = 10)
    # implement(case = 'seattle', alg = 'GBR', init_inf = [0], H_a = H_a, MU = MU, ibound = 20)
    implement(case = 'seattle', alg = 'BE', init_inf = init_inf, H_a = H_a, MU = MU)


def implement(alg = 'BP', init_inf = [0], H_a = 0.1, MU = 0.002, ibound=20, epsilon=.1):
    '''
        This runs BP, MF, GBR, and BE on a given GM
    '''
    file = './seattle/TractTravelRawNumbers.csv'
    file = './seattle/seattle_45sameArea_travel_numbers.csv'
    J = extract_data(file, MU=MU)
    # maxJ = np.max(J)
    # print(np.max(J), np.min(J))

    # create a complete graph GM
    model = generate_graphical_model(J, H_a)
    # print('model generated')

    # threshold the model
    # epsilon = .1 # percentage of Max J to drop edges
    # for fac in model.factors:
    #     if fac.log_values[0][0] < epsilon*maxJ and 'F' in fac.name:
    #         model.remove_factor(fac)


    # choose algorithm
    if alg == 'BP':
        init_inf = [ith_object_name('V',var) for var in init_inf]
        conditioned_on_init = condition_on_seeds_from(model, init_inf)
        # t1 = time.time()
        logZ = BeliefPropagation(conditioned_on_init).run(max_iter=1000, converge_thr=1e-4, damp_ratio=0.1)
        # t2 = time.time()
        CALIs = []
        for index in range(len(J)):
            if ith_object_name('V',index) not in init_inf:
                CALI = 2*logZ['marginals']['MARGINAL_V{}'.format(index)]-1
                CALIs.append(CALI[1])
            else:
                CALIs.append(+1)
        return CALIs
        # filename = "{}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format(alg, 'seattle', init_inf, H_a, MU)
    elif alg == 'MF':
        # condition_on_seeds_from(model, init_inf)
        init_inf = [ith_object_name('V',var) for var in init_inf]
        conditioned_on_init = condition_on_seeds_from(model, init_inf)
        logZ = MeanField(conditioned_on_init).run(max_iter=1000, converge_thr=1e-4)
        CALIs = []
        for index in range(len(J)):
            if ith_object_name('V',index) not in init_inf:
                CALI = 2*logZ['marginals']['MARGINAL_V{}'.format(index)]-1
                CALIs.append(CALI[1])
            else:
                CALIs.append(+1)
        return CALIs
    elif alg == 'GBR':
        # approximate
        CALIs = compute_marginals( model, (init_inf, H_a, MU, ibound), alg = alg)
        return CALIs
    elif alg == 'BE':
        # exact
        CALIs = compute_marginals( model, (init_inf, H_a, MU, ibound), alg = alg)
        return CALIs
    else:
        raise("Algorthim not defined")

def CALI_vs_MU(config):

    HS = list(config.keys())
    data = {}
    for alg in ['MF','BP','GBR20']:
        data[alg] = {}
        ibound=20
        if 'GBR' in alg: ibound = int(alg.replace('GBR',''))
        for H in HS:
            for inf in [4,74,82,111]:
                data[alg][str((H,inf))] = {}
                for MU in config[H]:
                    print("alg = {}, H = {}, inf = {}, MU = {}".format(alg, H, inf, MU))
                    t1 = time.time()
                    CALI = implement( alg = alg[:3], init_inf = [inf-1], H_a = H, MU = MU, ibound = ibound)
                    t2 = time.time()
                    print('time taken = {}'.format(t2-t1))
                    data[alg][str((H,inf))][MU] = CALI
                    plt.plot(MU*np.ones(len(CALI)), CALI, '*')
                plt.title(r"H_a={}, infected node {}".format(H, ith_object_name('V',inf)))
                plt.xlabel(r"$\mu$")
                plt.ylabel('CALI')
                plt.axis([config[H][0], config[H][-1], -1.1, 1.1])
                plt.savefig("./results/CALIvsMU_alg_{}_H={}_MUrange=[{},{}]_initinf={}.png".format(alg,H, config[H][0], config[H][-1], inf))
                plt.clf()

    with open("./results/CALI_vs_MU.json", 'w') as outfile:
        json.dump(data, outfile)

# an experiment
def runtime_vs_number_of_nodes():
    file = './seattle/TractTravelRawNumbers.csv'
    J = extract_data(file, MU=MU)
    print(np.max(J), np.min(J))

    # create a complete graph GM
    model = generate_graphical_model(J, H_a)
    print('model generated')


    CALI = implement(model, alg = 'BP', init_inf = [0], H_a = 0.1, MU = 0.002, ibound=10)



def thresholding_experiment():
    # files to extract data from
    # N is in [10,20,45,70,100]
    N=45
    logZs = []
    times = []
    taus = [float(i)/10 for i in range(10)]
    for tau in taus:
        file = './seattle/seattle_{}sameArea_travel_numbers.csv'.format(N)
        J = extract_data(file, MU=0.0002)
        maxJ = np.max(J)


        H_a = .1
        model = generate_graphical_model(J, H_a)
        edge_strengths = [fac.log_values[0][0] for fac in model.factors if 'F' in fac.name]
        # plot the degree distribution
        # degrees = [model.degree(var) for var in model.variables]
        # plt.subplot(2, 1, 1)
        # plt.plot(range(len(degrees)), degrees)
        # plt.title("degree distribution of complete graph")
        # plt.subplot(2, 1, 2)
        # sns.histplot([fac.log_values[0][0] for fac in model.factors if 'F' in fac.name])
        # plt.title("before")
        # plt.show()

        # thresholding
        num_rem = 0
        for fac in model.factors:
            if 'F' not in fac.name: continue
            edgeJ = fac.log_values[0][0]
            if edgeJ < tau*maxJ:
                model.remove_factor(fac)
                num_rem+=1
        print("{} number of edges have been dropped with tau= {}".format(num_rem, tau))
        # plot the degree distribution

        # degrees = [model.degree(var) for var in model.variables]
        # plt.subplot(2,1,1)
        # plt.plot(range(len(degrees)), degrees)
        # plt.title("degree distribution of thresholded graph")
        # plt.subplot(2,1,2)
        # sns.histplot([fac.log_values[0][0] for fac in model.factors if 'F' in fac.name])
        # plt.title("after")
        # plt.show()


        init_inf = [0]
        init_inf = [ith_object_name('V',var) for var in init_inf]
        conditioned_on_init = condition_on_seeds_from(model, init_inf)
        t1 = time.time()
        logZ = BucketRenormalization(conditioned_on_init, ibound=20).run()
        t2 = time.time()
        time_taken = t2-t1
        print("logZ = {}\ttime taken = {}".format(time_taken, logZ))
        logZs.append(logZ)
        times.append(time_taken)


    plt.plot(taus,logZs)
    plt.xlabel(r"$\tau$ threshold")
    plt.ylabel("Z value")
    plt.show()
    # CALI = implement(model, alg = 'GBR', init_inf = [0], H_a = H_a, MU = MU, ibound=10)
    # create GMs of different sizes from Amir's data
    # Create the thresholding experiment
    # define tau - a percentage of the maxJ (max strength of interaction)
    # drop edges in the GM with edge strength less than tau*maxJ
    # compute partition function and time it
# thresholding_experiment()


# extract data or a subset of the data
# J = extract_data_top10(MU)
# J = extract_data_top20('seattle', MU=MU)
# file = './seattle/seattle_20sameArea_travel_numbers.csv'
# file = './seattle/seattle_10sameArea_travel_numbers.csv'
# file = './seattle/TractTravelRawNumbers.csv'
# J = extract_data(file, MU=0.001)
#
# ibounds = [5*i for i in range(5,10)]
# for ibound in ibounds:
#     runtime = []
#     for i in range(10, 40, 3):
#         G = J[:i][:i]
#         print(i,np.max(G), np.min(G))
#         # create a complete graph GM
#         H_a = 0.1
#         model = generate_graphical_model(G, H_a)
#         print('model generated, size = {}'.format(len(G)))
#         alg = 'GBR'
#         t1 = time.time()
#         CALI = implement(model, alg = alg , init_inf = [0], H_a = 0.1, MU = 0.002, ibound=ibound)
#         t2 = time.time()
#         print(alg+str(ibound), len(CALI), t2-t1)
#         runtime.append(t2-t1)
#     plt.plot(range(10, 10+len(runtime)), runtime, label=alg+str(ibound))
# plt.title('GBR runtime as number of nodes grow')
# plt.xlabel('Number of nodes in GM')
# plt.ylabel('Runtime in seconds')
# plt.legend()
# plt.savefig('Runtime_VS_Nodenumber_GBR_large.png')
# plt.show()

# number of MU
N = 10
mu = 6e-4
mus = np.round(np.linspace(0,mu, N),5)
config = {key: mus for key in [0.02, 0.05, 0.1, 0.2, 0.5]}
config = {0.1: [0, 0.0025, 0.005, 0.0075, 0.01]}
CALI_vs_MU(config)

# How to Profile
# t1 = time.time()
# import cProfile
# cProfile.run("CALI = implement( alg = 'GBR', init_inf = [0], H_a = 0.1, MU = 6e-4, ibound = 20)")
# t2 = time.time()
# print(CALI, 'GBR', len(CALI), t2-t1)
