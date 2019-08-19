import pymisca.util as pyutil
import pandas as pd
import numpy as np

import plotly.offline as pyoff

def makeSliderPlot(dataset,
                  xCol,
                  yCol,
                  frameCol,
                  idCols=None,
                  textCol=None,
                  sizeCol=None,
                   colorCol=None,
                   sizeref=1.,
                    checkColor = 1,
                   copy = True,
                   mode=  'markers',
                  ):
    '''return a figure dictionary
    Adapted from: 
        https://stackoverflow.com/a/45825466/8083313
        https://plot.ly/python/gapminder-example/
'''
    if copy:
        dataset = dataset.copy()
    figure = {
        'data': [],
        'layout': {},
        'frames': [],
    }
    
    if idCols is None:
        dataset['id'] = dataset.index
        idCols=['id',]
        
    if sizeCol is None:
        dataset['size'] = 100.
    else:
        dataset['size'] = dataset[sizeCol]
    
    if 'color' not in dataset.keys():
        dataset['color'] = 'blue'
        
    if colorCol is None:
        dataset['name'] = 'Test'
        colorCol = 'name'
    else:
        dataset['name'] = dataset[colorCol] 

    dataset['text'] = dataset[textCol]
    dataset['x'] = dataset[xCol]
    dataset['y'] = dataset[yCol]

    figure['layout']['xaxis'] = {'title': xCol, 
                                 'type': 'log', 
                                 'autorange': True} #was not set properly
    figure['layout']['yaxis'] = {'title': yCol, 
                                 'autorange': True} #was not set properly
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['showlegend'] = True


    def getTraceDict(dfc, wrapper = lambda x:x.values,sizeref=sizeref):
        '''Get a trace from a dataframe
    '''
        if wrapper is not None:
            out = {}
            for key,val in dfc.iteritems():
    #             val = dfc[key]
                if key =='name':
                    val = str(val.values[0])
                else:
                    val  = wrapper(val)
                out[key] = val
            dfc = out

        data_dict = {            
                    'x': dfc['x'],
                    'y': dfc['y'],
                    'mode': mode,
                    'text': dfc['text'],
                    'marker': {
                        'sizemode': 'area',
                        'sizeref': sizeref,
                        'size': dfc['size'],
                        'color': dfc['color'],
                    },
                    'name': dfc['name']
                }
        return data_dict

    # frameKeys = [dataset]
    for i,(frameKey,dfc) in enumerate(dataset.groupby(frameCol)):
        frame = {'data': [], 'name': str(frameKey)}
        for colorKey,dfcc in dfc.groupby(colorCol):
            if checkColor:
                assert len(dfcc.color.unique())==1,'Mutiple color specified in the same traces!'
                
#             data_dict = dfcc.groupby(idCols,
#                                     ).apply(getTraceDict,wrapper=lambda x:x.values).tolist()
#             frame['data'].extend(data_dict)
            data_dict = getTraceDict(dfcc)
            frame['data'].append(data_dict)

        figure['frames'].append(frame) #this block was indented and should not have been.

        if i==0:
            figure['data'] = frame['data']
    
#     pointDF = dataset.set_index(colorCol,drop=0).copy()
#     pointDF.x= np.nan
#     pointDF.y= np.nan
    
#     pointDF = dataset.drop_duplicates(subset=colorCols,
#                                      ).set_index(colorCols,drop=0)    
#     pointDF = dataset.drop_duplicates(subset=idCols,
#                                      ).set_index(idCols,drop=0)    

#     pointDF = dataset.drop_duplicates(subset=idCols,
#                                      ).set_index(idCols,drop=0) 

#     frameZero =  pointDF.groupby(level=0).apply(getTraceDict,).tolist()
#     for trace in frameZero:
#         trace['x'] = [0.]
#         trace['y'] = [0.]
#     figure['data'] = frameZero



    ###### Slider And Updatemenus
    def frameKey2sliderDict(frameKey,):
        slider_step = {'args': [
            [frameKey],
            {'frame': {'duration': 10, 'redraw': True},
             'mode': 'immediate',
           'transition': {'duration': 100}}
         ],
         'label': frameKey,
         'method': 'animate'} 
        return slider_step

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Year:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 
                       'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': map(frameKey2sliderDict, 
                     sorted(dataset[frameCol].unique()),                
    #                  sorted(dataset[frameCol].unique())
                    )
    }


    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': 
                                    {'duration': 500, 
                                     'redraw': False},
                             'fromcurrent': True, 
                                    'transition': {'duration': 300, 
                                                   'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 
                                                'redraw': False},
                                      'mode': 'immediate',
                    'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]


    figure['layout']['sliders'] = [sliders_dict]

    return figure



if __name__=='__main__':
    pyoff.init_notebook_mode()

    dataBuffer = ''',country,continent,year,Country_code,Total_population,Life_satisfaction,GDP_per_capita
    62,Afghanistan,Asia,2008,AFG,29839994.0,3.723589897,1298.14315888
    63,Afghanistan,Asia,2009,AFG,30577756.0,4.401778221,1531.17399272
    64,Afghanistan,Asia,2010,AFG,31411743.0,4.75838089,1614.25500126
    65,Afghanistan,Asia,2011,AFG,32358260.0,3.83171916,1660.73985618
    66,Afghanistan,Asia,2012,AFG,33397058.0,3.782937527,1839.27357928
    67,Afghanistan,Asia,2013,AFG,34499915.0,3.572100401,1814.15582533
    167,Albania,Europe,2007,ALB,3169665.0,4.634251595,8447.88228539
    169,Albania,Europe,2009,ALB,3192723.0,5.485469818,9524.60981095
    170,Albania,Europe,2010,ALB,3204284.0,5.268936634,9927.13514733
    171,Albania,Europe,2011,ALB,3215988.0,5.867421627,10207.7006745
    172,Albania,Europe,2012,ALB,3227373.0,5.510124207,10369.7616592
    173,Albania,Europe,2013,ALB,3238316.0,4.550647736,10504.0930888
    242,Algeria,Africa,2010,DZA,35468208.0,5.46356678,12870.2162376
    243,Algeria,Africa,2011,DZA,35980193.0,5.317194462,12989.9549601
    244,Algeria,Africa,2012,DZA,36485828.0,5.604595661,13161.566464
    451,Angola,Africa,2011,AGO,19618432.0,5.589000702,5911.25433387
    452,Angola,Africa,2012,AGO,20162517.0,4.360249996,5998.63860099'''
    import StringIO
    dataset=pd.read_csv(StringIO.StringIO(dataBuffer))

    custom_colors = {
        'Asia': 'rgb(171, 99, 250)',
        'Europe': 'rgb(230, 99, 250)',
        'Africa': 'rgb(99, 110, 250)',
        'Americas': 'rgb(25, 211, 243)',
        #'Oceania': 'rgb(9, 255, 255)' 
        'Oceania': 'rgb(50, 170, 255)',
    }
    dataset['color'] = dataset.eval('continent.map(@custom_colors.get)')    

    testDict=  dict(
        xCol = 'GDP_per_capita',
        yCol = 'Life_satisfaction',
        textCol = 'country',
        # sizeCol = 'Total_population',
        idCols = ['country'],
        frameCol = 'year',
        colorCol = 'continent',
    )

    figure  = makeSliderPlot(dataset,**testDict)
    figDiv = pyoff.plot(figure, config={'scrollzoom': True},
                    show_link=0,auto_open=False, 
                    output_type='div')
    pyutil.printlines(figDiv,ofname='test__sliderPlot.html')
    #     pyoff.iplot(figure, config={'scrollzoom': True},)