# datos

La serie del grupo es la misma del miniproyecto 1.

| Campo | Valor |
|---|---|
| Archivo | `aep_mensual.csv` |
| Qué mide | Carga eléctrica media por hora de cada mes en la zona de American Electric Power (AEP), dentro de PJM, en MW |
| Fuente | Kaggle, [robikscube/hourly-energy-consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (Rob Mulla), archivo `AEP_hourly.csv`; datos originales de [PJM](https://www.pjm.com/) |
| Licencia | CC0: Public Domain |
| Frecuencia | mensual (agregada a partir de datos horarios) |
| Periodo | 2004-10 a 2018-07 (166 observaciones) |
| Columna de fecha | `mes` |
| Columna que se pronostica | `AEP_MW` |

## Cómo se construyó la serie mensual

El CSV es el mismo del miniproyecto 1, donde está el guion que lo arma
(`datos/preparar_aep.py`) a partir de la carga horaria original (121 269
horas, de octubre de 2004 a agosto de 2018):

- Las horas duplicadas por el cambio de horario de verano se promedian.
- Cada mes es el promedio de sus horas (`resample("MS").mean()`), así que el
  valor es la carga media por hora del mes, en MW.
- Se descartan los meses con menos de 672 horas (28 días): solo agosto de
  2018, que tiene 49 horas y quedaría sesgado.

`miniproyecto.py` lee directamente `aep_mensual.csv`, sin descargar nada.
