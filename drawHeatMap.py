from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

from matplotlib import rc

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def drawProbabilityHeatmap(passedFileName,tractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTitle,isShowZoomed=True):
    numPlots=len(subplotTitles)

    if numPlots<=5:
        fig, mainAxe = plt.subplots(figsize=(19.20, 4.6), constrained_layout=True)
        mainAxe.set_visible(False)
        fig.suptitle(figureTitle, fontsize=16)

        counter = 0
        padSize=0.01
        rightLegendPad=0.08
        subplotWidth=(1-padSize*(numPlots+1)-rightLegendPad)/numPlots
        for i in range(len(subplotTitles)):
            # ax = fig.add_subplot(gs[0, counter])
            ax = fig.add_axes([padSize * (counter + 1) + counter * subplotWidth, 0.05, subplotWidth, 0.8])
            # ax.axes.xaxis.set_visible(False)
            # ax.axes.yaxis.set_visible(False)
            ax.set_title(subplotTitles[i], fontsize=10)
            # axes.append(ax)

            ax.imshow(rawSeattleImage, interpolation="nearest")

            if isShowZoomed==True:
                # inset axes....
                axins = ax.inset_axes([0.01, 0.5, 0.47, 0.47])
                axins.imshow(rawSeattleImage, interpolation="nearest", origin="lower")
                # sub region of the original image
                x1, x2, y1, y2 = 450, 650, 400, 700
                axins.set_xlim(x1, x2)
                axins.set_ylim(y2, y1)
                axins.axes.xaxis.set_visible(False)
                axins.axes.yaxis.set_visible(False)
                axins.set_xticklabels('')
                axins.set_yticklabels('')

                ax.indicate_inset_zoom(axins)
                # ax.imshow(rawSeattleImage, cmap=rvb)

            for index in range(probabilities[counter].shape[1]):
                ax.add_patch(
                    Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=15,
                            height=15,
                            edgecolor='None',
                            facecolor=(probabilities[counter].iloc[0][index], 1 - probabilities[counter].iloc[0][index],0, 1),
                            linewidth=1))

            if isShowZoomed == True:
                for index in range(probabilities[counter].shape[1]):
                    axins.add_patch(
                        Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=11,
                                height=11,
                                edgecolor='None',
                                facecolor=(
                                probabilities[counter].iloc[0][index], 1 - probabilities[counter].iloc[0][index], 0, 1),
                                linewidth=1))

            for index in range(infectedUVCoords.shape[0]):
                actualPoint = [infectedUVCoords.iloc[index][1], infectedUVCoords.iloc[index][2]]
                Size = 14
                path = [
                    [actualPoint[0], actualPoint[1] - Size],
                    [actualPoint[0] - Size * 0.3, actualPoint[1] - Size * 0.3],
                    [actualPoint[0] - Size, actualPoint[1] - Size * 0.1],
                    [actualPoint[0] - Size * 0.2, actualPoint[1] + Size * 0.1],
                    [actualPoint[0] - Size * 0.8, actualPoint[1] + Size * 0.9],
                    [actualPoint[0], actualPoint[1] + Size * 0.6],
                    [actualPoint[0] + Size * 0.8, actualPoint[1] + Size * 0.9],
                    [actualPoint[0] + Size * 0.2, actualPoint[1] + Size * 0.1],
                    [actualPoint[0] + Size, actualPoint[1] - Size * 0.1],
                    [actualPoint[0] + Size * 0.3, actualPoint[1] - Size * 0.3],
                ]
                ax.add_patch(Polygon(path,
                                     color=(0, 0, 1, 1),
                                     linewidth=1))

            if isShowZoomed == True:
                for index in range(infectedUVCoords.shape[0]):
                    actualPoint = [infectedUVCoords.iloc[index][1], infectedUVCoords.iloc[index][2]]
                    Size = 12
                    path = [
                        [actualPoint[0], actualPoint[1] - Size],
                        [actualPoint[0] - Size * 0.3, actualPoint[1] - Size * 0.3],
                        [actualPoint[0] - Size, actualPoint[1] - Size * 0.1],
                        [actualPoint[0] - Size * 0.2, actualPoint[1] + Size * 0.1],
                        [actualPoint[0] - Size * 0.8, actualPoint[1] + Size * 0.9],
                        [actualPoint[0], actualPoint[1] + Size * 0.6],
                        [actualPoint[0] + Size * 0.8, actualPoint[1] + Size * 0.9],
                        [actualPoint[0] + Size * 0.2, actualPoint[1] + Size * 0.1],
                        [actualPoint[0] + Size, actualPoint[1] - Size * 0.1],
                        [actualPoint[0] + Size * 0.3, actualPoint[1] - Size * 0.3],
                    ]
                    axins.add_patch(Polygon(path,
                                            color=(0, 0, 1, 1),
                                            linewidth=1))

            counter = counter + 1

    elif numPlots>5 and numPlots<=12:
        fig, mainAxe = plt.subplots(figsize=(19.20, 10), constrained_layout=True)
        mainAxe.set_visible(False)
        fig.suptitle(figureTitle, fontsize=16)

        maxRows = 3
        veticalPad = 0.005
        if numPlots==6:
            maxColumn = 3
            padSize = 0.08
        elif numPlots==7:
            maxColumn = 4
            padSize = 0.02
        elif numPlots==8:
            maxColumn = 4
            padSize = 0.02
        elif numPlots==9:
            maxColumn = 5
            padSize = 0.01
        elif numPlots==10:
            maxColumn = 5
            padSize = 0.01
        elif numPlots==11:
            maxColumn = 4
            padSize = 0.08
        elif numPlots==12:
            maxColumn = 4
            padSize = 0.08

        counterP=0
        counterR = 0
        counterC=0
        rightLegendPad = 0.05
        topTiltePad=0.05
        lastRow=numPlots%maxColumn

        numRows=math.ceil(numPlots/maxColumn)
        subplotWidth = (1 - padSize * (maxColumn + 1) - rightLegendPad) / maxColumn
        subplotHeight = (1 - veticalPad * (numRows + 1) - topTiltePad) / numRows

        lastRowPad = ((1 - subplotWidth * (lastRow) - rightLegendPad)) / (lastRow+1)
        for i in range(len(subplotTitles)):
            print('___')
            print(counterR)
            print(counterC)
            print('___')
            # ax = fig.add_subplot(gs[counterR, counterC])
            if lastRow > 0:
                if counterR < numRows - 1:
                    ax = fig.add_axes([padSize * (counterC + 1) + counterC * subplotWidth,
                                       veticalPad * (counterR + 1) + (
                                               numRows - counterR - 1) * subplotHeight, subplotWidth,
                                       subplotHeight])
                else:
                    ax = fig.add_axes([lastRowPad * (counterC + 1) + counterC * subplotWidth,
                                       veticalPad * (counterR + 1) + (
                                               numRows - counterR - 1) * subplotHeight, subplotWidth,
                                       subplotHeight])
            else:
                ax = fig.add_axes([padSize * (counterC + 1) + counterC * subplotWidth,
                                   veticalPad * (counterR + 1) + (
                                           numRows - counterR - 1) * subplotHeight, subplotWidth,
                                   subplotHeight])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.set_title(subplotTitles[i], fontsize=10)
            # axes.append(ax)

            ax.imshow(rawSeattleImage, interpolation="nearest")

            if isShowZoomed == True:
                # inset axes....
                axins = ax.inset_axes([0.01, 0.5, 0.47, 0.47])
                axins.imshow(rawSeattleImage, interpolation="nearest", origin="lower")
                # sub region of the original image
                x1, x2, y1, y2 = 450, 650, 400, 700
                axins.set_xlim(x1, x2)
                axins.set_ylim(y2, y1)
                axins.axes.xaxis.set_visible(False)
                axins.axes.yaxis.set_visible(False)
                axins.set_xticklabels('')
                axins.set_yticklabels('')

                ax.indicate_inset_zoom(axins)

            # ax.imshow(rawSeattleImage, cmap=rvb)
            for index in range(probabilities[counterP].shape[1]):
                ax.add_patch(
                    Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=15,
                            height=15,
                            edgecolor='None',
                            facecolor=(probabilities[counterP].iloc[0][index], 1 - probabilities[counterP].iloc[0][index],0,1),
                            linewidth=1))

            if isShowZoomed == True:
                for index in range(probabilities[counterP].shape[1]):
                    axins.add_patch(
                        Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=11,
                                height=11,
                                edgecolor='None',
                                facecolor=(
                                probabilities[counterP].iloc[0][index], 1 - probabilities[counterP].iloc[0][index], 0,
                                1),
                                linewidth=1))

            for index in range(infectedUVCoords.shape[0]):
                actualPoint = [infectedUVCoords.iloc[index][1], infectedUVCoords.iloc[index][2]]
                Size = 14
                path = [
                    [actualPoint[0], actualPoint[1] - Size],
                    [actualPoint[0] - Size * 0.3, actualPoint[1] - Size * 0.3],
                    [actualPoint[0] - Size, actualPoint[1] - Size * 0.1],
                    [actualPoint[0] - Size * 0.2, actualPoint[1] + Size * 0.1],
                    [actualPoint[0] - Size * 0.8, actualPoint[1] + Size * 0.9],
                    [actualPoint[0], actualPoint[1] + Size * 0.6],
                    [actualPoint[0] + Size * 0.8, actualPoint[1] + Size * 0.9],
                    [actualPoint[0] + Size * 0.2, actualPoint[1] + Size * 0.1],
                    [actualPoint[0] + Size, actualPoint[1] - Size * 0.1],
                    [actualPoint[0] + Size * 0.3, actualPoint[1] - Size * 0.3],
                ]
                ax.add_patch(Polygon(path,
                                     color=(0, 0, 1, 1),
                                     linewidth=1))

            if isShowZoomed == True:
                for index in range(infectedUVCoords.shape[0]):
                    actualPoint = [infectedUVCoords.iloc[index][1], infectedUVCoords.iloc[index][2]]
                    Size = 12
                    path = [
                        [actualPoint[0], actualPoint[1] - Size],
                        [actualPoint[0] - Size * 0.3, actualPoint[1] - Size * 0.3],
                        [actualPoint[0] - Size, actualPoint[1] - Size * 0.1],
                        [actualPoint[0] - Size * 0.2, actualPoint[1] + Size * 0.1],
                        [actualPoint[0] - Size * 0.8, actualPoint[1] + Size * 0.9],
                        [actualPoint[0], actualPoint[1] + Size * 0.6],
                        [actualPoint[0] + Size * 0.8, actualPoint[1] + Size * 0.9],
                        [actualPoint[0] + Size * 0.2, actualPoint[1] + Size * 0.1],
                        [actualPoint[0] + Size, actualPoint[1] - Size * 0.1],
                        [actualPoint[0] + Size * 0.3, actualPoint[1] - Size * 0.3],
                    ]
                    axins.add_patch(Polygon(path,
                                            color=(0, 0, 1, 1),
                                            linewidth=1))

            counterC = counterC + 1
            if counterC >= maxColumn:
                counterR = counterR + 1
                counterC = 0

            counterP = counterP + 1

    else:
        print('not implemented')

    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap(
        [c('green'), c('red')])

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    divider = make_axes_locatable(mainAxe)
    cax = divider.append_axes('right', size='1%', pad='-1%')

    # cax1 = cax.append_axes("right", size="7%", pad="2%")
    # cb1 = colorbar(im1, cax=cax1)

    mpl.colorbar.ColorbarBase(cax, cmap=rvb,
                              norm=norm,
                              orientation='vertical')

    # divider = make_axes_locatable(mainAxe)
    # cax = divider.append_axes('right', size='10%', pad='0.5%')
    # cax.imshow(legend, cmap='bone')

    plt.tight_layout()
    plt.show()
    fig.savefig(passedFileName+".png")#AUTOMATICALLY SAVES THE IMAGE FILES IN RESULTS FOLDER

def renormalizeProbability(input,isNormalize=False,isThreeColumn=False,isTopNormalize=True,isBottomNormalize=True):
    output=pd.DataFrame()
    max_value=0
    min_value=1000000
    numbers=[]
    for i in range(input.shape[0]):
        if isThreeColumn == True:
            if input.iloc[i][0][1:].isnumeric():
                if input.iloc[i][3] < min_value:
                    min_value = input.iloc[i][3]
                if input.iloc[i][3] > max_value:
                    max_value = input.iloc[i][3]
                numbers.append(input.iloc[i][3])
        else:
            if input.iloc[i][0].isnumeric():
                if isinstance(input.iloc[i][1], np.float64):
                    if input.iloc[i][1] < min_value:
                        min_value = input.iloc[i][1]
                    if input.iloc[i][1] > max_value:
                        max_value = input.iloc[i][1]
                    numbers.append(input.iloc[i][1])
                else:
                    if input.iloc[i][1].isnumeric():
                        if input.iloc[i][1] < min_value:
                            min_value = input.iloc[i][1]
                        if input.iloc[i][1] > max_value:
                            max_value = input.iloc[i][1]
                            numbers.append(input.iloc[i][1])
                    else:
                        nums = input.iloc[i]['P'].strip('][').split(' ')
                        num = float(nums[0])
                        if num < min_value:
                            min_value = num
                        if num > max_value:
                            max_value = num
                        numbers.append(num)

    for i in range(input.shape[0]):
        if isThreeColumn == True:
            if input.iloc[i][0][1:].isnumeric():
                if isNormalize == True:
                    if isTopNormalize==True and isBottomNormalize==True:
                        if (max_value - min_value) != 0:
                            output[str(i)] = np.array([(numbers[i] - min_value) / (max_value - min_value)])
                        else:
                            output[str(i)] = np.array([numbers[i]])
                    elif isTopNormalize==True and isBottomNormalize==False:
                        output[str(i)] = np.array([(numbers[i]) / (max_value)])
                    elif isTopNormalize == False and isBottomNormalize == True:
                        output[str(i)] = np.array([(numbers[i]) - (min_value)])
                else:
                    output[str(i)] = np.array([numbers[i]])
        else:
            if input.iloc[i][0].isnumeric():
                if isNormalize == True:
                    if isTopNormalize == True and isBottomNormalize == True:
                        if (max_value - min_value) != 0:
                            output[str(i)] = np.array([(numbers[i] - min_value) / (max_value - min_value)])
                        else:
                            output[str(i)] = np.array([numbers[i]])
                    elif isTopNormalize == True and isBottomNormalize == False:
                        output[str(i)] = np.array([(numbers[i]) / (max_value)])
                    elif isTopNormalize == False and isBottomNormalize == True:
                        output[str(i)] = np.array([(numbers[i]) - (min_value)])
                else:
                    output[str(i)] = np.array([numbers[i]])

    return output

def getRevisedUVCoords(input,tractUVCoords,isThreeColumn=False):
    output = pd.DataFrame()
    for i in range(input.shape[0]):
        if isThreeColumn == True:
            if input.iloc[i][0][1:].isnumeric():
                output[str(i)] = tractUVCoords.iloc[int(input.iloc[i][0][1:])]
        else:
            if input.iloc[i][0].isnumeric():
                output[str(i)] = tractUVCoords.iloc[int(input.iloc[i][0])]


    return output.transpose()

def getInfectionUVCoords(initInfection,tractUVCoords):
    output = pd.DataFrame()
    for i in range(tractUVCoords.shape[0]):
        for j in range(len(initInfection)):
            if i == int(initInfection[j]):
                output[str(i)] = tractUVCoords.iloc[i]

    return output.transpose()

tractUVCoords = pd.read_csv('./seattle/seattle_UV_coordinates.csv')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES
rawSeattleImage=mpimg.imread('./seattle/SeattleRawImage2.png')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES


# initInfection=['81']
# H_a=['0.1']
# mu=['1e-05','0.0001','0.001','0.002','0.01','0.1']
# tau=['']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Marginal Probabilities initial infection=[' + initInfection[0] + ']'
# for i in initInfection:
#     for h in H_a:
#         for m in mu:
#             for t in tau:
#                 file_name='./results_BP_and_MF_comparison/BP_seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'.csv'
#                 subplotTitles.append('Initial infection='+i+r' $H_a$'+'=' + h + r' μ=' + m)
#                 temp=pd.read_csv(file_name)
#                 temp1 = renormalizeProbability(temp,isNormalize=False)
#                 probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_BP_and_MF_comparison/BP_seattle_heatmap_init_inf=[81]_H_a=[0.1]_MU_[1e-05,0.0001,0.001,0.002,0.01,0.1]_TAU_[]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle)
#
# initInfection=['81']
# H_a=['0.1']
# mu=['1e-05','0.0001','0.001','0.002','0.01','0.1']
# tau=['']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Marginal Probabilities initial infection=[' + initInfection[0] + ']'
# for i in initInfection:
#     for h in H_a:
#         for m in mu:
#             for t in tau:
#                 file_name='./results_BP_and_MF_comparison/MF_seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'.csv'
#                 subplotTitles.append('Initial infection=' + i + r' $H_a$' + '=' + h + r' μ=' + m)
#                 temp=pd.read_csv(file_name)
#                 temp1 = renormalizeProbability(temp)
#                 probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_BP_and_MF_comparison/MF_seattle_heatmap_init_inf=[81]_H_a=[0.1]_MU_[1e-05,0.0001,0.001,0.002,0.01,0.1]_TAU_[]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle)
#
#
#
# initInfection=['81']
# H_a=['0.1']
# mu=['0.0001','0.00014736842105263158','0.00019473684210526317','0.00024210526315789473','0.00028947368421052634','0.0003368421052631579','0.00038421052631578946','0.0004315789473684211','0.00047894736842105264']
# tau=['']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Marginal Probabilities initial infection=[' + initInfection[0] + ']'
# for i in initInfection:
#     for h in H_a:
#         for m in mu:
#             for t in tau:
#                 file_name='./results_transition/BP_seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'.csv'
#                 subplotTitles.append('Initial infection=' + i + r' $H_a$' + '=' + h + r' μ=' + str(round(float(m),5)))
#                 temp=pd.read_csv(file_name)
#                 temp1 = renormalizeProbability(temp)
#                 probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_transition/BP_seattle_heatmap_init_inf=[81]_H_a=[0.1]_MU_[1e-05,0.0001,0.001,0.002,0.01,0.1]_TAU_[]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle)
#
#
# initInfection=['0']
# H_a=['0.1']
# mu=['0.002']
# tau=['0','20','40','60','80','100']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Marginal Probabilities initial infection=[' + initInfection[0] + ']'
# for i in initInfection:
#     for h in H_a:
#         for m in mu:
#             for t in tau:
#                 file_name='./results_GBR/BP_seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'_TAU='+t+'.csv'
#                 subplotTitles.append('Initial infection=' + i + r' $H_a$' + '=' + h + r' μ=' + str(round(float(m),5)) + r' τ=' + t)
#                 temp=pd.read_csv(file_name)
#                 temp1 = renormalizeProbability(temp)
#                 probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_GBR/BP_seattle_heatmap_init_inf=[81]_H_a=[0.1]_MU_[0.002]_TAU_[0,20,40,60,80,100]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle)
#
#
# initInfection=['0']
# H_a=['0.1']
# mu=['1e-05','0.0001','0.01','0.1','0.5']
# tau=['120']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Marginal Probabilities initial infection=[' + initInfection[0] + ']'
# for i in initInfection:
#     for h in H_a:
#         for m in mu:
#             for t in tau:
#                 file_name='./results_GBR/BP_seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'_TAU='+t+'.csv'
#                 subplotTitles.append('Initial infection=' + i + r' $H_a$' + '=' + h + r' μ=' + str(round(float(m),5)) + r' τ=' + t)
#                 temp=pd.read_csv(file_name)
#                 temp1 = renormalizeProbability(temp)
#                 probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_GBR/BP_seattle_heatmap_init_inf=[0]_H_a=[0.1]_MU_[1e-05,0.0001,0.01,0.1,0.5]_TAU_[120]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle)
#
#
# initInfection=['0']
# H_a=['1.0','10.0']
# mu=['0.002']
# tau=['120']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Marginal Probabilities initial infection=[' + initInfection[0] + ']'
# for i in initInfection:
#     for h in H_a:
#         for m in mu:
#             for t in tau:
#                 file_name='./results_GBR/BP_seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'_TAU='+t+'.csv'
#                 subplotTitles.append('Initial infection=' + i + r' $H_a$' + '=' + h + r' μ=' + str(round(float(m),5)) + r' τ=' + t)
#                 temp=pd.read_csv(file_name)
#                 temp1 = renormalizeProbability(temp)
#                 probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_GBR/BP_seattle_heatmap_init_inf=[81]_H_a=[1.0,10.0]_MU_[0.002]_TAU_[120]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle)
#
#
# # results_GBR has 3 columns (different format)
# initInfection=['0']
# H_a=['0.0','0.1']
# mu=['0.001','0.0015','0.002']
# tau=['120.0']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Marginal Probabilities initial infection=[' + initInfection[0] + ']'
# for i in initInfection:
#     for h in H_a:
#         for m in mu:
#             for t in tau:
#                 file_name='./results_GBR/seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'_TAU='+t+'.csv'
#                 subplotTitles.append('Initial infection=' + i + r' $H_a$' + '=' + h + r' μ=' + str(round(float(m),5)) + r' τ=' + t)
#                 temp=pd.read_csv(file_name)
#                 temp1 = renormalizeProbability(temp,isThreeColumn=True,isNormalize=True,isBottomNormalize=False)
#                 probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords,isThreeColumn=True)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_GBR/seattle_heatmap_init_inf=[81]_H_a=[0.0,0.1]_MU_[0.001,0.0015,0.002]_TAU_[120]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle)
#
# # results_ibound has 3 columns (different format)
# initInfection=['81']
# H_a=['0.1']
# mu=['0.0001']
# tau=['']
# iBound=['10','12','15','17','18','19','20']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Marginal Probabilities initial infection=[' + initInfection[0] + ']'
# for ib in iBound:
#     for i in initInfection:
#         for h in H_a:
#             for m in mu:
#                 for t in tau:
#                     file_name='./results_ibound/GBR_ibound='+ib+'_seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'.csv'
#                     subplotTitles.append('IBound=' + ib + r' $H_a$' + '=' + h + r' μ=' + str(round(float(m),5)) + r' τ=' + t)
#                     temp=pd.read_csv(file_name)
#                     temp1 = renormalizeProbability(temp,isThreeColumn=True)
#                     probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords,isThreeColumn=True)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_ibound/GBR_ibound=[10,12,15,17,18,19,20]_seattle_heatmap_init_inf=[81]_H_a=[0.1]_MU_[0.0001]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle)
#
#
# # results_of_BP_MF_and_GBR (GBR) has 3 columns (different format)
# initInfection=['81']
# H_a=['0.1']
# mu=['0.0001']
# tau=['']
# iBound=['']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Different Algorithms Marginal Probabilities initial infection=[' + initInfection[0] + ']'
#
# file_name='./results_of_BP_MF_and_GBR/BP_seattle_marg_prob_init_inf=[81]_H_a=0.1_MU=0.0001.csv'
# subplotTitles.append('BP algorithm' + r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp)
# probabilities.append(temp1)
#
# file_name='./results_of_BP_MF_and_GBR/GBR_seattle_marg_prob_init_inf=[81]_H_a=0.1_MU=0.0001.csv'
# subplotTitles.append('GBR algorithm' + r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp,isThreeColumn=True)
# probabilities.append(temp1)
#
# file_name='./results_of_BP_MF_and_GBR/MF_seattle_marg_prob_init_inf=[81]_H_a=0.1_MU=0.0001.csv'
# subplotTitles.append('MF algorithm' + r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp)
# probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_of_BP_MF_and_GBR/Different_Algorithms_seattle_heatmap_init_inf=[81]_H_a=[0.1]_MU_[0.0001]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle)
#
#
#
# # results_of_BP_MF_and_GBR (GBR) has 3 columns (different format)
# initInfection=['0']
# H_a=['0.1']
# mu=['0.0005']
# tau=['']
# iBound=['10','20']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Different Algorithms Marginal Probabilities initial infection=[' + initInfection[0] + ']'
#
# file_name='./results_20_node_comparison/BP_seattle_marg_prob_init_inf=[0]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('BP algorithm' + r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp)
# probabilities.append(temp1)
#
# file_name='./results_20_node_comparison/GBR_ibound=10_seattle_marg_prob_init_inf=[0]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('GBR algorithm iBound=10'+ r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp,isThreeColumn=True)
# probabilities.append(temp1)
#
# file_name='./results_20_node_comparison/GBR_ibound=20_seattle_marg_prob_init_inf=[0]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('GBR algorithm iBound=20'+ r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp,isThreeColumn=True)
# probabilities.append(temp1)
#
# file_name='./results_20_node_comparison/MF_seattle_marg_prob_init_inf=[0]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('MF algorithm' + r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp)
# probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_20_node_comparison/Different_Algorithms_seattle_heatmap_init_inf=[0]_H_a=[0.1]_MU_[0.0005]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle,isShowZoomed=False)
#
#
#
# # results_of_BP_MF_and_GBR (GBR) has 3 columns (different format)
# initInfection=['0']
# H_a=['0.1']
# mu=['0.0005']
# tau=['']
# iBound=['10','20']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Different Algorithms Marginal Probabilities initial infection=[' + initInfection[0] + ']'
#
# file_name='./results_20_node_comparison_testing/BP_seattle_marg_prob_init_inf=[0]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('BP algorithm' + r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp)
# probabilities.append(temp1)
#
# file_name='./results_20_node_comparison_testing/GBR_ibound=10_seattle_marg_prob_init_inf=[0]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('GBR algorithm iBound=10'+ r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp,isThreeColumn=True,isNormalize=True,isBottomNormalize=False)
# probabilities.append(temp1)
#
# file_name='./results_20_node_comparison_testing/GBR_ibound=20_seattle_marg_prob_init_inf=[0]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('GBR algorithm iBound=20'+ r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp,isThreeColumn=True,isNormalize=True,isBottomNormalize=False)
# probabilities.append(temp1)
#
# file_name='./results_20_node_comparison_testing/MF_seattle_marg_prob_init_inf=[0]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('MF algorithm' + r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp)
# probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_20_node_comparison_testing/Different_Algorithms_seattle_heatmap_init_inf=[0]_H_a=[0.1]_MU_[0.0005]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle,isShowZoomed=False)
#
#
# # results_of_BP_MF_and_GBR (GBR) has 3 columns (different format)
# initInfection=['0']
# H_a=['0.1']
# mu=['0.0005']
# tau=['']
# iBound=['10','20']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Different Algorithms Marginal Probabilities initial infection=[' + initInfection[0] + ']'
#
# file_name='./results_20_node_comparison_testing/BP_seattle_marg_prob_init_inf=[0]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('BP algorithm' + r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp)
# probabilities.append(temp1)
#
# file_name='./results_20_node_comparison_testing1/GBR_ibound=10_seattle_marg_prob_init_inf=[0]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('GBR algorithm iBound=10'+ r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp,isThreeColumn=True,isNormalize=True,isBottomNormalize=False)
# probabilities.append(temp1)
#
# file_name='./results_20_node_comparison_testing1/MF_seattle_marg_prob_init_inf=[0]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('MF algorithm' + r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp)
# probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_20_node_comparison_testing1/Different_Algorithms_seattle_heatmap_init_inf=[0]_H_a=[0.1]_MU_[0.0005]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle,isShowZoomed=False)
#
#
# # results_ibound has 3 columns (different format)
# initInfection=['0']
# H_a=['0.1']
# mu=['0.0001','0.0002','0.0003','0.0004','0.0005','0.0006']
# tau=['']
# iBound=['']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Marginal Probabilities initial infection=[' + initInfection[0] + ']'
# for ib in iBound:
#     for i in initInfection:
#         for h in H_a:
#             for m in mu:
#                 for t in tau:
#                     file_name='./results_20_node_comparison_testing1/BE_seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'.csv'
#                     subplotTitles.append('Initial infection=' + i + r' $H_a$' + '=' + h + r' μ=' + m)
#                     temp=pd.read_csv(file_name)
#                     temp1 = renormalizeProbability(temp,isThreeColumn=True,isNormalize=True,isBottomNormalize=False)
#                     probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords,isThreeColumn=True)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_20_node_comparison_testing1/BE_seattle_heatmap_init_inf=[0]_H_a=[0.1]_MU_[0.0001,0.0002,0.0003,0.0004,0.0005,0.0006]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle,isShowZoomed=False)
#
#
#
# # results_of_BP_MF_and_GBR (GBR) has 3 columns (different format)
# initInfection=['81']
# H_a=['0.1']
# mu=['0.0005']
# tau=['']
# iBound=['10']
# probabilities=[]
# temp=[]
# subplotTitles=[]
# figureTiltle='Different Algorithms Marginal Probabilities initial infection=[' + initInfection[0] + ']'
#
# file_name='./results_30_node_comparison/BP_seattle_marg_prob_init_inf=[81]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('BP algorithm' + r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp)
# probabilities.append(temp1)
#
# file_name='./results_30_node_comparison/GBR_ibound=10_seattle_marg_prob_init_inf=[81]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('GBR algorithm iBound=10'+ r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp,isThreeColumn=True,isNormalize=True,isBottomNormalize=False)
# probabilities.append(temp1)
#
# file_name='./results_30_node_comparison/MF_seattle_marg_prob_init_inf=[81]_H_a=0.1_MU=0.0005.csv'
# subplotTitles.append('MF algorithm' + r' $H_a$' + '=' + H_a[0] + r' μ=' + str(round(float(mu[0]),5)))
# temp=pd.read_csv(file_name)
# temp1 = renormalizeProbability(temp)
# probabilities.append(temp1)
#
# revisedTractUVCoords=getRevisedUVCoords(temp,tractUVCoords)
# infectedUVCoords=getInfectionUVCoords(initInfection,tractUVCoords)
# test_name = "./results_30_node_comparison/Different_Algorithms_seattle_heatmap_init_inf=[81]_H_a=[0.1]_MU_[0.0005]"
# drawProbabilityHeatmap(test_name,revisedTractUVCoords,infectedUVCoords,rawSeattleImage,initInfection,probabilities,subplotTitles,figureTiltle)