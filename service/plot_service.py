import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as clr

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from analysis import threshold


def getSignificanceSymbol(pvalue, oneStar=0.05, twoStars=0.01, threeStars=0.001):
    symbol = ''
    if (pvalue<=oneStar):
        symbol = '*'
    if (pvalue<=twoStars):
        symbol = '**'
    if (pvalue<=threeStars):
        symbol = '***'
    return symbol

def createFontSizes(regular=24):
    medium = 0.8 * regular
    small = 0.7 * regular
    tiny = 0.6 * regular
    return regular, medium, small, tiny


def createArialNarrowFont():
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rc('font',family='Arial')
    return {'fontname':'Arial', 'stretch' : 'condensed'}


def extractColorsFromColormap(n=10, colormap='jet'):
    cmap = cm.get_cmap(colormap)
    norm = mpl.colors.Normalize(vmin=0, vmax=n-1) 
    return [cmap(norm(ind)) for ind in range(n)] 



def shiftToMaxSurvivalTime(survivalTime, survivalEvent, maxSurvivalTime):
    if ((survivalTime is None) or (survivalEvent is None)):
        return np.nan, np.nan
    shiftedTime = survivalTime
    shiftedEvent = survivalEvent
    if (survivalTime>maxSurvivalTime):
        shiftedTime = maxSurvivalTime
        shiftedEvent = 0.0
    return shiftedTime, shiftedEvent   


def add_axe_annotations_for_kaplan_meier_plot(ax, title, max_survival_time):
    
    regular, medium, small, tiny = createFontSizes(regular=24)
    arialNarrowFont = createArialNarrowFont()
    
    ax.set_title(title, fontsize=regular, **arialNarrowFont) 
    
    L = ax.legend(fontsize=small)
    plt.setp(L.texts, **arialNarrowFont)
    
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0.0 - 0.05*max_survival_time, max_survival_time + 0.05*max_survival_time])
    
    nticks = 5
    step = np.ceil(max_survival_time/nticks)
    # step = 20 * np.ceil(max_survival_time/100.0)
    xticks = np.arange(0, max_survival_time + step, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, **arialNarrowFont)
    ax.set_xlabel('Time', fontsize=regular, **arialNarrowFont)
    
    yticks = [0.01*t for t in range(0, 101, 20)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, **arialNarrowFont)
    ax.set_ylabel('Survival probability', fontsize=regular, **arialNarrowFont)
    ax.tick_params(axis='both', labelsize=small)


def plot_individual_survival(data, gene, group_threshold, survival_data, duration_col='time', event_col='event', max_survival_time=None):
    
    cox = pd.DataFrame(index=data.index)
    cox[gene] = data[gene]
    
    if max_survival_time is None:
        max_survival_time = survival_data[duration_col].max()
    
    for id_sample in data.index:
        shifted_time = survival_data.loc[id_sample, duration_col]
        shifted_event = survival_data.loc[id_sample, event_col]
        shifted_time, shifted_event = shiftToMaxSurvivalTime(shifted_time, shifted_event, max_survival_time)
        cox.loc[id_sample, 'time'] = shifted_time
        cox.loc[id_sample, 'event'] = shifted_event
    
    cox['group'] = 'Low'
    cox['group_bin'] = 0.0
    cox.loc[cox[gene]>group_threshold[gene], 'group'] = 'High'
    cox.loc[cox[gene]>group_threshold[gene], 'group_bin'] = 1.0
     
    group_colors = {'Low': 'royalblue', 'High': 'crimson'} 
     
    kmf = KaplanMeierFitter()
    
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    
    for group in ['Low', 'High']:
        cox_group = cox[cox['group']==group]
        label = group + ' (n=' + str(cox_group.shape[0]) + ')'
        kmf.fit(cox_group['time'], cox_group['event'], label=label)
        kmf.plot(ax=ax, ci_show=False, show_censors=True, color=group_colors[group], linewidth=3)
    
    # Logrank test
    logrank = multivariate_logrank_test(cox['time'], cox['group_bin'], cox['event'])
    logrank_text = 'logrank p-value = ' + '{:.1e}'.format(logrank.p_value) + ' ' + getSignificanceSymbol(logrank.p_value)
    
    # Cox model    
    cph = CoxPHFitter()
    cox_bin = cox[['group_bin', 'time', 'event']]
    cph.fit(cox_bin, duration_col='time', event_col='event', show_progress=False)
    # cox_pvalue = cph.summary.p['group_bin']
    cox_hr = cph.summary['exp(coef)']['group_bin']
    hr_text = 'HR between groups = ' + '{:.2f}'.format(cox_hr)
    
    title = gene + '\n' + logrank_text + '\n' + hr_text
    add_axe_annotations_for_kaplan_meier_plot(ax, title, max_survival_time)

    return fig


def getGroupName(groupNumberGenes, n):
    for groupName in groupNumberGenes:
        nmin = groupNumberGenes[groupName][0]
        nmax = groupNumberGenes[groupName][1]
        if (n>=nmin and n<=nmax):
            return groupName

def getGroupColors(groupNumberGenes):
    nbGroups = len(groupNumberGenes)
    groupColors = {'P1' : 'royalblue', 'P2' : 'black', 'Not expressed' : 'royalblue', 'Expressed' : 'black'}
    if (nbGroups==3):
        groupColors = {'P1' : 'royalblue', 'P2' : 'crimson', 'P3' : 'black'}
    if (nbGroups==4):
        groupColors = {'P1' : 'royalblue', 'P2' : 'orange', 'P3' : 'crimson', 'P4' : 'black'}
    if (nbGroups==5):
        groupColors = {'P1' : 'royalblue', 'P2' : 'orange', 'P3' : 'crimson', 'P4' : 'darkviolet', 'P5' : 'black'}
    if (nbGroups==6):
        groupColors = {'P1' : 'cyan', 'P2' : 'royalblue', 'P3' : 'orange', 'P4' : 'crimson', 'P5' : 'darkviolet', 'P6' : 'black'}
    if (nbGroups==7):
        groupColors = {'P1' : 'cyan', 'P2' : 'royalblue', 'P3' : 'orange', 'P4' : 'red', 'P5' :'crimson', 'P6' : 'darkviolet', 'P7' : 'black'}
    return  groupColors


def generateGroupLabel(groupNumberGenes, group):
    nmin = groupNumberGenes[group][0]
    nmax = groupNumberGenes[group][1]
    if (nmin<nmax):
        return str(nmin) + '-' + str(nmax)
    else:
        return str(nmin)


def plot_combined_survival(data, genes, prognosis_groups, group_threshold, survival_data, duration_col='time', event_col='event', max_survival_time=None):
    
    cox = pd.DataFrame(index=data.index)
    cox[genes] = data[genes]
    
    if max_survival_time is None:
        max_survival_time = survival_data[duration_col].max()
    
    for id_sample in data.index:
        shifted_time = survival_data.loc[id_sample, duration_col]
        shifted_event = survival_data.loc[id_sample, event_col]
        shifted_time, shifted_event = shiftToMaxSurvivalTime(shifted_time, shifted_event, max_survival_time)
        cox.loc[id_sample, 'time'] = shifted_time
        cox.loc[id_sample, 'event'] = shifted_event
    
        n = 0
        for gene in genes:
            if (cox.loc[id_sample, gene]) > group_threshold[gene]:
                n = n + 1
        cox.loc[id_sample, 'n'] = n
        cox.loc[id_sample, 'group'] = getGroupName(prognosis_groups, n)

    # Logrank test
    logrank = multivariate_logrank_test(cox['time'], cox['group'], cox['event'])  
    logrank_text = 'logrank p-value = ' + '{:.1e}'.format(logrank.p_value) + ' ' + getSignificanceSymbol(logrank.p_value)
    
    # Cox model    
    cph = CoxPHFitter()
    cox_n = cox[['n', 'time', 'event']]
    cph.fit(cox_n, duration_col='time', event_col='event', show_progress=False)
    cox_pvalue = cph.summary.p['n']
    cox_hr = cph.summary['exp(coef)']['n']
    cox_text = 'cox p-value = ' + '{:.1e}'.format(cox_pvalue) + ' ' + getSignificanceSymbol(cox_pvalue)
    hr_text = 'HR between groups = ' + '{:.2f}'.format(cox_hr)
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    groupColors = getGroupColors(prognosis_groups)

    kmf = KaplanMeierFitter()
    for group in sorted(prognosis_groups.keys()):
        selection = cox[cox['group']==group]
        nSamples = selection.shape[0]
        labelText = generateGroupLabel(prognosis_groups, group) + ' (n=' + str(nSamples) +')'
        kmf.fit(selection['time'], selection['event'], label=labelText)
        kmf.plot(ax=ax, ci_show=False, show_censors=True, color=groupColors[group], linewidth=3)
        
    title = logrank_text + '\n'  + cox_text + '\n' + hr_text
    add_axe_annotations_for_kaplan_meier_plot(ax, title, max_survival_time)
    
    return fig
    
'''    
     
    group_colors = {'Low': 'royalblue', 'High': 'crimson'} 
     
    kmf = KaplanMeierFitter()
    
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    
    regular, medium, small, tiny = createFontSizes(regular=24)
    arialNarrowFont = createArialNarrowFont()
    
    for group in ['Low', 'High']:
        cox_group = cox[cox['group']==group]
        label = group + ' (n=' + str(cox_group.shape[0]) + ')'
        kmf.fit(cox_group['time'], cox_group['event'], label=label)
        kmf.plot(ax=ax, ci_show=False, show_censors=True, color=group_colors[group], linewidth=3)
    
    # Logrank test
    logrank = multivariate_logrank_test(cox['time'], cox['group_bin'], cox['event'])
    logrank_text = 'logrank p-value = ' + '{:.1e}'.format(logrank.p_value) + ' ' + getSignificanceSymbol(logrank.p_value)
    
    # Cox model    
    cph = CoxPHFitter()
    cox_bin = cox[['group_bin', 'time', 'event']]
    cph.fit(cox_bin, duration_col='time', event_col='event', show_progress=False)
    # cox_pvalue = cph.summary.p['group_bin']
    cox_hr = cph.summary['exp(coef)']['group_bin']
    hr_text = 'HR between groups = ' + '{:.2f}'.format(cox_hr)
    
    ax.set_title(gene + '\n' + logrank_text + '\n' + hr_text, fontsize=regular, **arialNarrowFont) 
    
    L = ax.legend(fontsize=small)
    plt.setp(L.texts, **arialNarrowFont)
    
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0.0 - 0.05*max_survival_time, max_survival_time + 0.05*max_survival_time])
    
    nticks = 5
    step = np.ceil(max_survival_time/nticks)
    # step = 20 * np.ceil(max_survival_time/100.0)
    xticks = np.arange(0, max_survival_time + step, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, **arialNarrowFont)
    ax.set_xlabel('Time in months', fontsize=regular, **arialNarrowFont)
    
    yticks = [0.01*t for t in range(0, 101, 20)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, **arialNarrowFont)
    ax.set_ylabel('Overall survival', fontsize=regular, **arialNarrowFont)
    ax.tick_params(axis='both', labelsize=small)
    
    return fig
'''