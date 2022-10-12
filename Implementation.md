# Implementation Notes


- Summary of differences between simulations for hindcast (truth) simulation used to generate data and simulations used in data assimilation forecasting experiments:




|                               	| Data Assimilation (Forecasts) 	|       Truth (Hindcasts) 	|     Location             |
|-------------------------------	|-------------------------------	|-------------------------	|----------------------    |
| Domain                        	| Gulf Of Mexico                	| Western North Atlantic  	|    fort.14               |
| Avg Mesh Elements size ($km^2$) 	| 98                            	| 1.34                    	|    fort.                 |
| Time step (s)                 	| 10                            	| 1                       	|    fort.15               |
| Wind Field                    	| Dynamic Holland               	| OWI                     	|    fort.22               |
| Bottom Friction Formulation   	| Chezy                         	| Hybrid                  	|    fort.15 (NOLIBF)      |


Full run time: 9 days (216 hours)
Hours prior tolandfall  run times:
- 12 hours (204 hours total runtime)
- 24 hours (192 hours total runtime)
- 36 hours (180 hours total runtime)
- 48 hours (168 hours total runtime)
