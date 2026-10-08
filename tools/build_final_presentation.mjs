/** Editable research deck from audited evidence. Requires Codex Artifact Tool runtime. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const root=process.cwd(), build=path.join(root,'.presentation-build');
const skill=process.env.PRESENTATION_SKILL_DIR, runtime=process.env.ARTIFACT_TOOL_ROOT;
if(!skill||!runtime)throw Error('Set PRESENTATION_SKILL_DIR and ARTIFACT_TOOL_ROOT to the provided Codex runtime paths');
process.env.RUNTIME_NODE_MODULES ??= path.resolve(runtime,'../..');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'dist/artifact_tool.mjs')).href);
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont({fontFamily:'Noto Sans'});
const d=JSON.parse(await fs.readFile(path.join(root,'artifacts/presentation_data.json'),'utf8'));
await fs.mkdir(build,{recursive:true});await fs.mkdir(path.join(root,'artifacts/final'),{recursive:true});
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={dark:'#183B32',green:'#0C825D',blue:'#52718A',muted:'#62706C',orange:'#B16A2E',bg:'#FAFCF9'};
const num=x=>Math.round(x).toLocaleString('ru-RU');
function txt(s,text,x,y,w,h,size=26,bold=false,color=C.dark){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=text;a.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};return a;}
function slide(title,n,note=''){let s=p.slides.add();s.background.fill=C.bg;txt(s,title,64,40,1152,94,44,true);txt(s,`${n} / 21`,1130,668,86,25,17,false,C.muted);if(note)txt(s,note,64,619,1040,60,19,false,C.muted);return s;}
function table(s,values,y=180,height=330,widths,fontSize=24,left=64,width=1152){const t=s.tables.add({rows:values.length,columns:values[0].length,left,top:y,width,height,values,columnWidths:widths});t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({style:'solid',fill:'#D1DDD5',width:0.7});for(let i=0;i<values.length;i++)for(let j=0;j<values[0].length;j++){let c=t.getCell(i,j);c.fill=i===0?C.dark:i%2?'#F0F6F1':'#FFFFFF';c.text.style={typeface:font,fontSize,color:i===0?'#FFFFFF':C.dark,bold:i===0};}return t;}
function chart(s,type,categories,series,pos,extra={}){let c=s.charts.add(type,{position:pos,categories,series:series.map(q=>({...q,values:q.values.map(v=>v==null?null:Number(v.toFixed(3)))})),hasLegend:series.length>1,legend:{position:'bottom',textStyle:{typeface:font,fontSize:21}},chartFill:C.bg,plotAreaFill:C.bg,xAxis:{textStyle:{typeface:font,fontSize:21}},yAxis:{numberFormatCode:'0.0',textStyle:{typeface:font,fontSize:21},majorGridlines:{fill:'#DCE6DD',width:0.8}},...extra});applyPresentationChartFont(c,{fontFamily:font});return c;}
function notes(s,text){s.speakerNotes.textFrame.setText(text);}
let s=slide('Расходы муниципалитетов\nи сигналы изменений',1);
txt(s,'СберИндекс · задача №2',64,188,930,50,28,false,C.green);
txt(s,'Автор проекта: NikitaMeshDS',64,245,930,44,25);
txt(s,'Прогноз на 1, 3, 6 и 12 месяцев',64,329,930,65,36,true);
txt(s,'Сезонная модель с поправкой роста\nи обнаружение изменений методом EWMA',64,413,930,97,28);
const repoLabel='github.com/NikitaMeshDS/sberindex-task2';
const repoLink=txt(s,repoLabel,64,531,965,40,23,false,C.green);repoLink.text.get(repoLabel).link={uri:d.jury_guide.repository,isExternal:true};
const viewerLabel='nikitameshds.github.io/sberindex-task2';
const viewerLink=txt(s,viewerLabel,64,579,965,37,23,false,C.green);viewerLink.text.get(viewerLabel).link={uri:d.jury_guide.dashboard,isExternal:true};
s.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,'docs/assets/repository-qr.png'))),contentType:'image/png',alt:'QR: публичный репозиторий для сдачи на GitHub',fit:'contain',position:{left:1000,top:190,width:208,height:208}});
txt(s,'GitHub: открытый доступ',987,410,233,76,18,false,C.muted);
txt(s,'Данные 2023–2024 · готовые материалы для самостоятельной проверки',64,640,1010,30,21,false,C.muted);
notes(s,'Author handle from repositoryowner, no additionalteamidentityclaimed. QR and native hyperlink target the public single-root submission repository. Viewer hyperlink targets GitHub Pages. Private research repository history is not published. Sources: jury_guide.json andreport. Retrospectivearchive2024, no independent2025facts.');

s=slide('Итоги и карта семи критериев',2,'¹ К одному Prophet с профилем; интервалы условны на 2024. ² Ложные тревоги / 100 МО-месяцев на синтетике.');
txt(s,'MAE −42 / −23 / −28 / −56%¹ · EWMA F1 0,647',64,151,1152,56,31,true,C.green);
const criterionProofs=['Формула и объяснение словами','10 моделей, 4 горизонта и выбор по MAE','8 методов, точность и ложные тревоги','Сравнение, преимущества и ограничения','Два канала: прогноз и поиск изменений','MAE, R² и объяснение каждой метрики','Примеры МО и воспроизведение'];
const criterionLabels=['Методология','Прогноз','Детекторы','Готовые модели','Новости','Метрики','Интерпретация'];
table(s,[['Критерий','Вес','Доказательство','Слайды / отчёт'],...d.jury_guide.criteria.map((x,i)=>[criterionLabels[i],x.weight,criterionProofs[i],`${x.slides} / §${x.report_sections}`])],208,397,[240,80,570,262],24);
notes(s,'Criterionweightsfromcompetition total100percent. Exactgainvsfixedpooled41.5339/23.3389/28.4866/55.7996; displayrounded42/23/28/56. h12bestdisabledcontrol54.53 retainedreport. EWMAevaluationonsetF1.646650, FAR2/4320*100=.0463. Foundationcompareactualsavedtables, notbenchmarkbreadthscore. Newsquoteclassificationnotspendingcausality. Slides19numbering andreportsectionmapping verifiedinpackagingaudit.');

s=slide('Данные и временная проверка',3,'Исторические версии данных и сроки выпуска расходов неизвестны. Расчёт использует сценарий лага 0.');
const dataBlocks=[['СберИндекс','2 190 ID','24 месяца · 6 категорий\nСредние расходы жителя, руб.'],['География и Росстат','2 075 МО','Положительная история 2023\nРост зарплат: выпуск 27.12.2023'],['Временная проверка','1 / 3 / 6 / 12','Горизонты в месяцах\nПрофили строим по 2023 году']];
dataBlocks.forEach(([label,value,body],i)=>{let x=64+i*394;txt(s,label,x,174,370,50,29,true);txt(s,value,x,254,370,78,47,true,C.green);txt(s,body,x,358,365,117,25);});
txt(s,'Факты для оценки: 2031 / 2031 / 2029 / 2023 МО\nВнешние источники соединяем по дате доступности и территории',64,510,1152,95,27);
notes(s,'Sources: data/consumption.parquet; reports/full_cohort_review/coverage.csv; data_sources.json; growthsource officialpublication. Cohorteligibility determined onlyfrom2023. Geography2024snapshot availability unverified. Rawpanel2190IDs vs2075eligible vsvariable2031scored. No independent new-year test.');

s=slide('Одна модель для четырёх горизонтов',4,'17,2% — номинальный рост зарплат за октябрь 2023; опубликован 27.12.2023, затем заморожен.');
s.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,'artifacts/formulas/forecast.png'))),contentType:'image/png',alt:'Прогноз: уровень × отношение сезонных профилей × поправка роста',fit:'contain',position:{left:64,top:159,width:1100,height:105}});
txt(s,'Берём текущий уровень, сдвигаем по сезонному профилю;\nпри переходе через Новый год добавляем годовой рост',64,270,1152,100,28);
txt(s,'Профиль = 0,5 × общий профиль + 0,5 × профиль региона\nk — число переходов года; g = 0,172',64,388,1152,90,26);
txt(s,'Выбор одной модели по MAE ранних целей h1 / h3 / h6\nh12 не участвует в выборе; поздние цели показаны отдельно',64,516,1152,90,25);
notes(s,'Sources: src/sberindex/forecasting/growth_bridge_review.py; configs/growth_bridge_review.json; reports/growth_bridge_review/selection.json. https://rosstat.gov.ru/storage/mediabank/osn-11-2023.pdf; https://rosstat.gov.ru/central-news?page=52&per_page=10&print=1. Source available from28Dec2023 scenario, search-index extraction, immutability unverified. Choice rule defined after inspection2024; no independent holdout.');

s=slide('Модель обходит Prophet на всех горизонтах',5,'MAE: руб./жителя. 2031 / 2031 / 2029 / 2023 МО; 12 / 10 / 7 / 1 дат. R² h1 — изменения внутри МО. HGB h12: нет обучаемых примеров.');
const modelNames=[['seasonal_naive','Seasonal naive'],['prophet_disabled','Prophet: базовый'],['prophet_yearly3','Prophet: Fourier 3'],['prophet_pooled_profile','Prophet + профиль'],['seasonal_pooled','Сезонный профиль'],['hierarchy05','Иерархия 0,5'],['direct_lags','Direct HGB'],['chronos2_univariate','Chronos-2'],['chronos2_profile','Chronos-2 + ковариата'],['hierarchy05_growth_bridge','Иерархия 0,5 + рост']];
const cr=d.expansion.fullpanel;
const mt=[['Модель','h1','h3','h6','h12','R² h1'],...modelNames.map(([m,label])=>[label,...[1,3,6,12].map(h=>{let x=cr.find(q=>q.model===m&&q.horizon===h);return x?.MAE==null||(m==='direct_lags'&&h===12)?'—':num(x.MAE);}),cr.find(q=>q.model===m&&q.horizon===1).R2_within_MO.toFixed(3)])];
const horizons=[1,3,6,12];
const chosenMAE=horizons.map(h=>cr.find(x=>x.model==='hierarchy05_growth_bridge'&&x.horizon===h).MAE);
const bestProphetMAE=horizons.map(h=>Math.min(...cr.filter(x=>x.model.startsWith('prophet_')&&x.horizon===h).map(x=>x.MAE)));
txt(s,'MAE, руб./жителя',64,149,535,32,23,true);
chart(s,'bar',horizons.map(h=>`h${h}`),[{name:'Выбранная модель',values:chosenMAE,valuesFormatCode:'0',dataLabelOverrides:chosenMAE.map((v,idx)=>({idx,text:num(v),position:'outEnd',textStyle:{typeface:font,fontSize:19}})),fill:C.green},{name:'Лучший Prophet',values:bestProphetMAE,valuesFormatCode:'0',dataLabelOverrides:bestProphetMAE.map((v,idx)=>({idx,text:num(v),position:'outEnd',textStyle:{typeface:font,fontSize:19}})),fill:C.blue}],{left:64,top:195,width:580,height:343},{barOptions:{direction:'column',grouping:'clustered',gapWidth:70},yAxis:{min:0,max:3500,numberFormatCode:'0',textStyle:{typeface:font,fontSize:20},majorGridlines:{fill:'#DCE6DD',width:0.8}},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:19},},legend:{position:'bottom',textStyle:{typeface:font,fontSize:20}}});
txt(s,'Полная панель: десять моделей',668,149,548,32,23,true);
const compactNames=['Seasonal naive','Prophet: базовый','Prophet: Fourier 3','Prophet + проф.','Сезонный','Иерархия 0,5','Direct HGB','Chronos-2','Chronos + ков.','Иерархия + рост'];
const compactMT=mt.map((row,i)=>i===0?row:[compactNames[i-1],...row.slice(1)]);
const selectedTable=table(s,compactMT,199,337,[182,74,74,74,74,74],18,664,552);
for(let j=0;j<6;j++){selectedTable.getCell(10,j).fill='#D9EEE2';selectedTable.getCell(10,j).text.style={typeface:font,fontSize:18,color:C.dark,bold:true};}
txt(s,`Контроль на графике: лучший Prophet отдельно для каждого горизонта.
h12: базовый Prophet (2906), в слайде 6 единый Prophet с профилем (2989).`,64,551,1152,59,22,false,C.green);
notes(s,'Sources: reports/foundation_expansion_review/fullpanel_summary.csv/fullpanel_matched_predictions.parquet; same60700pairs full eligiblepanel, 2031/2031/2029/2023MO. Earlier251MOtable is superseded here. MainfullcohortselectedMAE932/1272/1371/1321 in report; This table and grouped-bar chart use the full matched panel. Bar control is the minimum MAE of the three Prophet variants separately at each horizon, including disabled at h12 (2906) instead of fixed profile (2989). directHGBh12 onlyfallback excluded. Covariate rowchronos2_profile notgroupedmodel; separate grouped2177to1589sourcefoundation_covariate_review. WithinMO R2pooledSSE overwithinactualvariation, notmeanlocalR2, notYoYR2. h12 denominator0, no R2.');

s=slide('Неопределённость выигрыша прогноза',6,'Интервалы выигрыша не содержат ноль. Узкое семейство h1/h3 исследовательское и добавлено после просмотра результатов.');
const ci=d.expansion.uncertainty.filter(x=>x.cluster==='region_code'&&x.comparator==='prophet_pooled_profile');
table(s,[['Горизонт','Снижение MAE','95% по регионам'],...ci.map(x=>[`${x.horizon} мес.`,`${x.reduction_pct.toFixed(1)}%`,`${x.reduction_low.toFixed(1)}–${x.reduction_high.toFixed(1)}%`])],170,280,[280,340,532],28);
txt(s,'Единый контроль: Prophet с сезонным профилем\n2500 повторных выборок МО и регионов целиком',64,472,1152,65,25);
txt(s,'По датам, чувствительность Holm h1/h3: p = 0,034 / 0,307\nШесть тестов: p h1 = 0,102; h12: временной тест не определён',64,548,1152,63,23,false,C.green);
notes(s,'Source: forecast_uncertainty_review/cluster_intervals.csv, temporal_tests.csv. PaireddatebalancedMAE; slide uses fixed pooled comparator all horizons; h12 gain55.80 CIregion50.52to60.97; besttestedh12disabled54.53 separately in report. Crosssectional CIconditional2024, nottemporal CI; postselection tests exploratory. DM h1T12/h3T10, HACBartlett h-1, HLN,tT-1,Holm6tests none5%; requested posthoc two pooled-horizon family h1 p=.03406, h3 p=.30684, not preregistered. h6 numericpossiblebutwithheld byauditminT=max(8,2h+1), h12T1 undefined.');

s=slide('Как читать метрики качества',7,'MAE усредняем по датам. WAPE считаем по всем парам; это разные способы взвешивания ошибок.');
const selectedH1=d.expansion.fullpanel.find(x=>x.model==='hierarchy05_growth_bridge'&&x.horizon===1);
table(s,[['Метрика','Результат','Что означает'],
 ['MAE / WAPE',`932 руб. / ${selectedH1.WAPE_pct.toFixed(2)}%`,'Средняя абсолютная ошибка на жителя. WAPE: сумма ошибок / сумма расходов.'],
 ['R² внутри МО','0,819 на h1','Квадратичная ошибка на 82% ниже сравнения со средним фактом каждого МО.'],
 ['Отрицательный R²','Naive, Fourier 3, HGB','R² < 0: модель хуже среднего значения МО в оценочном окне.'],
 ['95% интервал','Выигрыш выше нуля','Региональная выборка поддерживает выигрыш на 2024. Независимый временной тест не заменяет.'],
 ['F1 / FAR / задержка','0,647 / 0,046 / 0,54','F1: баланс точности и полноты начала. FAR: ложные тревоги / 100 МО-месяцев. Задержка: по найденным.'],
 ['Интервальный штраф IS','3237 → 4362 руб.','Ширина плюс штраф за промах. Меньше лучше; ×1,5 здесь проигрывает.'],
 ['Нагрузка тревог','0,597 → 0,075','Эпизоды на 100 МО-месяцев: очередь проверки меньше в 8 раз. Реальные ложные тревоги неизвестны.']],150,451,[240,260,652],20);
notes(s,'Source fullpanel_summary.csv selectedh1 MAE932.296 WAPE3.040075 R2within.818567 =1-SSE/sum((actual-meanactualperMO)^2). No claim of variance of firstdifferences or82percent of all changes correctlypredicted. Mean comparator uses evaluationfacts and is descriptive,notoperationalforecast. WAPE ratio rows3.04notdirectMAEmeanratio due dateweighting. SyntheticF1/FAR/delay andrealmonitoringburden different populations, delay420matched excludes300misses. CI conditional2024,nottemporal.');

s=slide('Результат устойчив к росту 7–17%',8,'Сохранённый архив 2024; не независимое доказательство переноса. Все переходы года — из декабря 2023.');
const getScenario=g=>d.sensitivity.filter(x=>x.subset==='all'&&Math.abs(x.growth_rate-g)<1e-8).sort((a,b)=>a.horizon-b.horizon);
table(s,[['Поправка','h1','h3','h6','h12'],...[[0,'Без роста'],[.0748,'ИПЦ: 7,48%'],[.172,'Зарплаты: 17,2%']].map(([g,label])=>[label,...getScenario(g).map(x=>num(x.MAE))])],180,265,[360,198,198,198,198]);
txt(s,'Любая ставка 7–17% снижает MAE к лучшему проверенному Prophet\nна всех четырёх горизонтах: проверены границы выпуклой ошибки',64,478,1152,88,27,true,C.green);
txt(s,'ИПЦ лучше на h1; зарплатная поправка — на h3 / h6 / h12',64,582,1152,40,24);
notes(s,'Sources: presentation_claim_checks/evidence.json andgrid_check.csv. Convexbound: k0/1 forecastsaffineing, date-balancedMAE convex, endpointmaxat7%/17% strictlybelowbesttestedProphetallh. This isfixed diagnosticnot newgselection. NovCPI7.48released8Dec2023 admissible; annualDecCPI7.4released12Jan2024 excludedfromasofslide, retainedreportexpost. Groupcohortfull60700pairs, not251modeltable.');

s=slide('Перенос модели на другие категории',9,'Одна формула и рост 17,2%, без перенастройки. Снижение MAE к заранее закреплённому Prophet с профилем; отрицательное значение — проигрыш.');
const cats=[...new Set(d.expansion.categories.map(x=>x.category))];
table(s,[['Категория','h1, %','h3, %','h6, %','h12, %'],...cats.map(cat=>[cat,...[1,3,6,12].map(h=>{let rows=d.expansion.categories.filter(x=>x.category===cat&&x.horizon===h),a=rows.find(x=>x.model==='hierarchy05_growth_bridge'),b=rows.find(x=>x.model==='prophet_pooled_profile');return (100*(1-a.MAE/b.MAE)).toFixed(1);})])],157,335,[420,183,183,183,183],25);
txt(s,'Единого преимущества на всех категориях нет\nProphet настроен по общей сумме; категории отдельно не настраивались',64,536,1152,73,26,true,C.green);
notes(s,'Source: category_growth_review/full_pooled_summary.csv. Exactpairswithineachcategory, eligibility2023only, fixedg17.2 andweight.5, frozen pooledProphet asprimarynewcategorycontrol chosenpriorresults. Totalall3Prophetcached separately; no percategory tuning. Marketplace losesh3/h6/h12; g0ablationfullcategoryresults retained.');

s=slide('Год истории: сезонная модель сильнее готовых',10,'Одинаковые 7532 пары: 251 / 251 / 252 / 252 МО. MAE в рублях. Современное предобучение ограничивает ретроспективные выводы.');
const fmNames=[['seasonal_pooled','Сезонный контроль'],['hierarchy05_growth_bridge','Выбранная модель'],['timesfm25_zero_shot','TimesFM 2.5'],['timesfm25_seasonal_ratio','TimesFM · отношение'],['moirai2_zero_shot','Moirai 2'],['moirai2_seasonal_ratio','Moirai · отношение'],['tirex_zero_shot','TiRex'],['tirex_seasonal_ratio','TiRex · отношение'],['bolt_base_univariate','Bolt · исходный'],['bolt_finetune_epoch1','Bolt · одна эпоха']];
const foundationTable=table(s,[['Модель','h1','h3','h6','h12'],...fmNames.map(([m,label])=>[label,...[1,3,6,12].map(h=>num(d.expansion.foundation.find(x=>x.model===m&&x.horizon===h).MAE))])],143,395,[400,188,188,188,188],23);
for(let j=0;j<5;j++){foundationTable.getCell(2,j).fill='#D9EEE2';foundationTable.getCell(2,j).text.style={typeface:font,fontSize:23,color:C.dark,bold:true};}
txt(s,'Работают без дообучения на МО с историей\nНа МО без истории не проверялись',64,555,1152,60,24,false,C.green);
notes(s,'Source: foundation_expansion_review/summary.csv, bolt_epoch1_training.json,provenance.json. New3modelsall3modes exactly7532keys; causalratio, fixedensemble50/50 nottuned. FullBoltcontextJanJun2023/trainJulSep/validationOctDec, fixed1epoch17batches,no2024selection. Mainnewcheckpointdateslater2024 contaminationunknown. TiRexupstreamGitdynamicpadding; TimesFM2.5torchCPU notMLX3.0.');

s=slide('Foundation-модели: преимущества и ограничения',11,'Контроли различаются: уменьшение ошибки внутри семейства не означает превосходства выбранной сезонной модели.');
txt(s,'Преимущества в экспериментах',64,174,560,80,30,true,C.green);
txt(s,'Без дообучения на новых МО с историей\n\nTimesFM: сезонное отношение\n−45% MAE на h1 к zero-shot\n\nBolt: одна эпоха обучения −22% на h1\n\nГрупповой Chronos-2: −27% на h1',64,272,565,282,26);
txt(s,'Границы применения',665,174,550,80,30,true,C.green);
txt(s,'Год истории даёт один сезонный цикл\n\nОшибка h12 в 3,7–9,3 раза выше\nвыбранной модели на той же выборке\n\nСостав предобучения неизвестен\n\nПрименение без истории не проверяли',665,272,550,282,26);
notes(s,'Sources foundation_expansion_review/summary.csv andfoundation_covariate_review/summary.csv. TimesFM 1-1146.59/2078.02=.4482; Bolt epoch1 1-1885.67/2416.25=.2196; groupedChronos MAE2176.66to1589 from251MO h1 earliercontrolledreview. h12range foundationvariantsincludingseasonalratio4966.17/selected1357.02=3.6596 to12612.81/1357.02=9.2945. NewMO meanszero-shot with history,notcoldstartzeroobservations. No claim benefit universally orallmodeslonghorizons.');

s=slide('Реальные МО: факт, прогноз и интервал',12,'Скользящие прогнозы h1. Иллюстративные 80% интервалы: ошибки трёх прошлых месяцев, масштаб 2023; лаг 0, гарантий покрытия нет.');
txt(s,'Факт — тёмная линия · прогноз — зелёная · светлые линии — границы интервала',64,146,1152,44,25);
d.comparison_cases.cases.forEach((c,i)=>{
  const x=64+i*393, months=c.months.filter(q=>q.lower!=null), cats=months.map(q=>q.target.slice(5));
  txt(s,c.name,x,204,375,55,29,true);
  const ymax=Math.ceil(Math.max(...months.map(q=>Math.max(q.actual,q.upper)))/1000)*1000;
  const pos={left:x,top:270,width:375,height:288};
  const ymin=Math.max(0,Math.floor(Math.min(...months.map(q=>Math.min(q.actual,q.predicted??q.actual,q.lower)))*0.96/1000)*1000);
  const axes={hasLegend:false,xAxis:{textStyle:{fontSize:18,typeface:font}},yAxis:{min:ymin,max:ymax,numberFormatCode:'0',textStyle:{fontSize:18,typeface:font}}};
  chart(s,'line',cats,[{name:'Верхняя граница 80%',values:months.map(q=>q.upper),line:{fill:'#9DBEAC',width:1.4}},{name:'Нижняя граница 80%',values:months.map(q=>q.lower),line:{fill:'#9DBEAC',width:1.4}},{name:'Факт',values:months.map(q=>q.actual),fill:'none',line:{fill:C.dark,width:2.5}},{name:'Прогноз h1',values:months.map(q=>q.predicted),fill:'none',line:{fill:C.green,width:2.5}}],pos,{...axes,lineOptions:{smooth:false}});
  if(i===2){const aprilX=x+80;s.shapes.add({geometry:'rect',position:{left:aprilX,top:300,width:2,height:215},fill:C.orange,line:{fill:'none',width:0}});txt(s,'Апрель: паводок',aprilX+5,263,275,29,18,false,C.orange);}
  txt(s,i===0?'Крупный город':i===1?'Район с низкими расходами':'Паводок: кейс проверки',x,568,375,44,23,false,C.muted);
});
notes(s,'Source:presentation_comparison_cases/case_months.csv/calibration.csv/evidence.json. Ufa fixednamedcapital, low2023expensemunicipaldistrictfixedby2023proxy notpopulation norerrors, Orskfixedstresscase. Allcurves selectedhierarchy05_growth_bridge samecachedpoints. AprDecplot onlysinceJanMarinsufficientprevious3months. Forecastissuetargetstart assumedlag0 includespreviousmonthknownthen. Bandsnewposthocillustrative80rankglobalnormalizedresidualquantile, notoldblendbands orformalcoverageguarantee; serialdependence. Tables sourceincludesJanMarwithoutbands. Values rub/consumer, eachyaxisownscale, notsingleyearaheadforecast.');

s=slide('EWMA: лучший онлайн F1 и минимальный FAR',13,'Ложные тревоги — на синтетике, на 100 МО-месяцев. Задержка только по найденным началам; пропущенные события исключены.');
const drows=d.verified_claims.detectors.rows.filter(x=>x.split==='evaluation');
const order=[['ewma','EWMA'],['cusum','CUSUM'],['cusum_spike','CUSUM + разовый'],['spike','Разовый порог'],['rolling_3m','Среднее 3 месяца'],['bocpd','BOCPD'],['pelt','PELT · офлайн'],['kernelcpd','Kernel · офлайн']];
const detectorLabels=['EWMA','CUSUM','CUSUM +' ,'Разовый порог','Среднее 3 мес.','BOCPD','PELT (офлайн)','Kernel (офлайн)'];
const labelPositions=['left','top','right','left','left','right','left','right'];
const detectorSeries=order.map(([m],i)=>{const x=drows.find(q=>q.method===m), offline=i>=6;return{name:detectorLabels[i],xValues:[Number(x.null_FAR_per100_mo_months.toFixed(6))],values:[x.onset_f1],fill:offline?'#919A96':i===0?C.green:C.blue,line:{fill:offline?'#919A96':i===0?C.green:C.blue,width:1},marker:{symbol:offline?'diamond':'circle',size:i===0?14:10},dataLabelOverrides:[{idx:0,text:i===3?'Порог':detectorLabels[i],position:labelPositions[i],showValue:false,textStyle:{typeface:font,fontSize:19,bold:i===0,fill:offline?'#62706C':C.dark}}]};});
chart(s,'scatter',[],detectorSeries,{left:64,top:176,width:730,height:365},{hasLegend:false,scatterOptions:{style:'marker'},xAxis:{min:0,max:0.17,majorUnit:0.04,numberFormatCode:'0.00',title:{text:'FAR: ложные тревоги / 100 МО-месяцев',textStyle:{typeface:font,fontSize:20}},textStyle:{typeface:font,fontSize:20}},yAxis:{min:0,max:0.85,majorUnit:0.2,numberFormatCode:'0.0',title:{text:'F1 начала шока',textStyle:{typeface:font,fontSize:20}},textStyle:{typeface:font,fontSize:20},majorGridlines:{fill:'#DCE6DD',width:0.8}},dataLabels:{showValue:false,textStyle:{typeface:font,fontSize:19}}});
const dt=table(s,[['Метод','F1','FAR','Мес.'],...order.map(([m],i)=>{let x=drows.find(q=>q.method===m);return[(i===2?'CUSUM + разовый':detectorLabels[i]).replace(' (офлайн)','*'),x.onset_f1.toFixed(3),x.null_FAR_per100_mo_months.toFixed(3),i>=6?'—':x.mean_onset_delay_months.toFixed(2)];})],184,337,[190,75,75,70],19,806,410);
for(let j=0;j<4;j++){dt.getCell(1,j).fill='#D9EEE2';dt.getCell(1,j).text.style={typeface:font,fontSize:19,color:C.dark,bold:true};}
for(let i=7;i<9;i++)for(let j=0;j<4;j++)dt.getCell(i,j).fill='#E5E8E7';
txt(s,'Серые ромбы и *: офлайн, используют будущий ряд',806,531,410,48,19,false,C.muted);
txt(s,`EWMA: F1 0,647 при FAR 0,046. Задержка 0,54 мес.
420 начал найдены, 300 пропущены.`,64,551,722,61,23,false,C.green);
notes(s,'Source: reports/presentation_claim_checks/evidence.json, short_history/comparison.csv/series_metrics.csv. Selectionfixedbeforeevaluation, eligibilitybestonlineonsetF1notfastest. Offlinetwoshadedrowsfullfuturesequencecannotcompareonline delay. FARdenominator4320nullmomonths/2alarms EWMA. EWMArecall420/720=.583. Delayonlytp±1monthmatchedfirstonsetmax(0,pred-onset), missesexcluded. Othermethodsfastdelaydoesnotimplyhighrecall.');

s=slide('Региональный остаток выделяет локальный сигнал',14,'Весь доступный ряд сравнен с прежними порогами. Частота тревог — нагрузка мониторинга, не ложные срабатывания.');
s.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,'artifacts/formulas/peer_residual.png'))),contentType:'image/png',alt:'Логарифмический остаток МО минус медиана остатков остальных МО региона',fit:'contain',position:{left:64,top:148,width:1152,height:105}});
txt(s,'Только другие МО региона и факты того же доступного месяца\nНе менее 5 соседей; пропуски остаются неизвестными',64,275,1152,93,28);
const eb=d.burden.filter(x=>x.method==='ewma');
table(s,[['Остаток','МО-месяцы','Эпизоды EWMA','На 100 месяцев'],...['own','peer'].map(k=>{let x=eb.find(a=>a.signal===k);return[k==='own'?'Собственный':'Относительно региона',num(x.observed_months),num(x.alert_episodes),x.episodes_per100_observed_months.toFixed(3)]})],398,194,[390,230,270,262]);
notes(s,'Source: reports/peer_residual_review/monitoring_burden.csv; protocol. Leave-one-outregionalmedian nofuturetimeinputs. Snapshot2024nottruehistoricalgeography. Thresholdssyntheticfrozen, noOrsktuning. Localchannelcanremovecommonregionalshock; retain ownchannel separately. Reducedburden doesnotprovebetterrecall.');

s=slide('Орск: локальный сигнал на фоне региона',15,'Орск минус Оренбург, годовой рост: +7,18 п.п. в мае–июле. Выбранное описательное окно; причинный эффект не установлен.');
chart(s,'line',d.orsk.map(x=>x.target.slice(5)),[{name:'Орск минус Оренбург, п.п.',values:d.orsk.map(x=>x.orsk_minus_orenburg_pp),line:{fill:C.green,width:3}}],{left:64,top:162,width:1152,height:345},{lineOptions:{smooth:false},yAxis:{min:0,max:11,numberFormatCode:'0.0',textStyle:{fontSize:21,typeface:font}},xAxis:{title:'Месяц 2024',textStyle:{fontSize:21,typeface:font}}});
txt(s,'Май: положительный региональный остаток — 16-е место из 2012 МО\nEWMA: 0,0345 < порога 0,0743; автоматической тревоги нет',64,528,1152,80,27,false,C.green);
notes(s,'Source: reports/peer_residual_review/case_yoy.csv, alarm_panel.csv/REPORT.md. Orsk1673 Orenburg1665. Paireddifference notdifference-in-differencescausality design, Orenburgalsoaffected. All4frozenmethods no2024alarm. Paymentsandrepairplausiblehypothesis; municipalMFC announcesrentsupportMay2inMay6post, notindividualpaymentslinktoaggregateexpense.');

s=slide('Новости входят в прогноз и поиск изменений',16,'Прогноз: исходный реестр; детекция: расширенный корпус. ×1,5 — гипотеза, не калиброванная по новостям.');
const newsSteps=['Официальный\nтекст','Qwen\nJSON события','Проверка\nточной цитаты','Территория\nID / регион','Доступность\nпубликация +1'];
newsSteps.forEach((v,i)=>{let x=64+i*236;let a=s.shapes.add({geometry:'rect',position:{left:x,top:155,width:207,height:88},fill:'#EDF3EE',line:{fill:'#A5C2B1',width:1}});a.text=v;a.text.style={typeface:font,fontSize:22,color:C.dark,bold:true,autoFit:'none'};if(i<4)s.shapes.add({geometry:'rightArrow',position:{left:x+210,top:190,width:24,height:16},fill:C.green,line:{fill:'none',width:0}});});
const nb=d.recovery_news_burden, baseline=nb.find(x=>x.policy==='baseline'), conditioned=nb.find(x=>x.policy==='quoted_events_factor075');
txt(s,`${d.recovery_news.combined_distinct_input_documents} текстов; ${d.recovery_news.combined_accepted_unique_urls} фрагментов с цитатой; ${d.recovery_news.combined_unique_municipalities} ID МО`,64,261,1152,44,27,true,C.green);
const ni=d.news_intervals.filter(x=>x.scope==='available_news_subset');
const nbase=ni.find(x=>x.policy==='baseline'), nwide=ni.find(x=>x.policy==='news_radius_x1_5');
txt(s,'Новости в прогнозе',64,329,550,38,29,true);
txt(s,`Новость известна к дате прогноза\nТочка прежняя; радиус интервала ×1,5\nПокрытие: ${nbase.covered}/${nbase.pairs} → ${nwide.covered}/${nwide.pairs} (94,4%)\nШирина: ${num(nbase.mean_width)} → ${num(nwide.mean_width)} руб.\nШтраф IS: ${num(nbase.mean_interval_score)} → ${num(nwide.mean_interval_score)} руб.\nВыигрыша нет; 18 пар, 6 дат`,64,378,565,166,22);
txt(s,'Новости в детекторе',665,329,550,38,29,true);
txt(s,`Только явно названные МО\nПорог × 0,75 на два месяца\n${baseline.episodes} → ${conditioned.episodes} эпизодов; ${d.recovery_news.municipalities_in_detector_panel.length} МО в панели\nСравнение с прежним порогом\nУпоминание события не доказывает шок`,665,382,550,161,23);
txt(s,'Прогноз: python -m sberindex.external.news_interval_review',64,554,1152,25,18,false,C.green);
txt(s,'Детекция: python -m sberindex.external.news_body_recovery',64,586,1152,25,18,false,C.green);
notes(s,'Forecast: reports/news_interval_review/summary.csv news_pairs.csv audit.json configs/news_interval_review.json; identical18pairs9MO6dates, originalMCHSregistry, availableatforecast, prior3month80pct residualintervals,2023scale, lag0. Pointunchanged, fixedfactor1.5posthoc; coverage17/18 unchanged width2852.307to4278.461 IS3236.925to4361.646(alpha.2), notimprovement; notappliedtomainmodel. Negative +/-3pct preservedreport. Detection: expandedrecoveredquotedcorpus,differentcoverage fromforecast, source news_body_recovery_20261007 audit/monitoringburden. Commands requirefullarchiveandinstalledpackage. Metadatahistoricalversionunconfirmed. ISsourcehttps://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618 section2.2.');

s=slide('Архитектура решения',17,'Конфигурации, исходные снимки и индивидуальные результаты сохранены вместе с происхождением и SHA256.');
const labels=['Данные\nи версии','Признаки\nсезонность / рост','Модель\n4 горизонта','Детектор\nМО / регион','Аналитик\nпроверка сигнала'];
labels.forEach((v,i)=>{const x=64+i*236;const a=s.shapes.add({geometry:'rect',position:{left:x,top:182,width:207,height:112},fill:i===4?'#D9EEE2':'#EDF3EE',line:{fill:'#A5C2B1',width:1}});a.text=v;a.text.style={typeface:font,fontSize:24,color:C.dark,bold:true,autoFit:'none'};if(i<4)s.shapes.add({geometry:'rightArrow',position:{left:x+210,top:231,width:24,height:16},fill:C.green,line:{fill:'none',width:0}});});
txt(s,'Новости → доступность и территория → контекст для аналитика',64,365,1152,66,31,false,C.green);
txt(s,'Что получает аналитик',64,465,1152,48,30,true);
txt(s,'Прогноз расходов · величина отклонения · подтверждающие источники\nОчередь эпизодов для проверки; интерфейс показывает сохранённые результаты',64,528,1152,79,26);
notes(s,'Sources: docs/ARCHITECTURE.md; existingCLIscientificpipeline. Nativeeditablearchitecture. Analystinterfaceisproposedrole/output notclaimeddeployedUI. Selectedpointforecastnonewsnumericfeature; eventcontextonlyuntilcoveragequalified. Controlleddatejoin beforeactualsignalinterpretation. CommandsavailableREADME, intentionallynotonslides.');

s=slide('2025: две версии роста сохранены для проверки',18,'Расчёт 07.10.2026 без муниципальных фактов 2025. Правило предложено после изучения 2024 и сообщения об агрегате 2025.');
txt(s,'В каждом файле: 8 092 прогноза · 2 023 одинаковых МО',64,162,1152,66,31,true,C.green);
table(s,[['Версия','Правило роста','Рост (медиана)'],['Исходная','Рост зарплат Росстата','17,2%'],['Новый кандидат','Медиана годового роста региона за 3 месяца','14,79%']],255,160,[240,695,217],24);
txt(s,`Недостаточно годовой истории → документированный макроисточник
Медианный прогноз кандидата ниже на 2,05%; диапазон −6,01…+3,16%`,64,446,1152,80,25);
txt(s,`2025: проверка покажет поведение при замедлении роста
Оба файла, входы и код имеют SHA256; прежний ансамбль также сохранён`,64,546,1152,64,25,true,C.green);
notes(s,'Sources: selected_model_holdout_20261007/freeze_manifest.json and adaptive_growth_holdout_20261007/freeze_manifest.json, growth_rates.csv, change_from_fixed.csv. Both8092rows2023MO origDec2024+h1/3/6/12. Candidate g median ofthree monthly regional medians ofMOYoY, >=10 valid2023eligiblepairs eachmonth. Otherwise lastdocumentedavailable macro17.2; no updatedmacroseriesclaimed. Profile2023 weight.5 unchanged. Median14.79427%, medianforecastchange-2.05267%, individualrange-6.01104to+3.161909%. Both2024diagnosticsidentical because allcrossingsorigDec2023 fallback. Candidateproposedafter2024review anduser2025aggregatecontext, notblind or preregistered. No2025municipalfacts; noindependentmetrics. Lag0assumption. Originalfreezeand75/25preserved.');

s=slide('Границы доверия к результатам',19);
txt(s,'Архив 2024 многократно изучен',64,170,1152,48,30,true);
txt(s,'Независимого временного теста нет. На h12 одна целевая дата.\nИнтервалы по регионам условны на этих данных.',64,236,1152,88,27);
txt(s,'Ограничения детекции и внешних источников',64,366,1152,48,30,true);
txt(s,'В реальной панели шоки не размечены; частота ложных тревог неизвестна.\nРасходные интервалы примеров иллюстративны.\nИсторическая доступность источников не подтверждена.\nПредсказание будущих реальных шоков не доказано.',64,434,1152,162,25,false,C.muted);
notes(s,'Unchangedscientificlimitsfromreportsection9 andpreviousslide16. NegativecaseOrsknoalarm retained. Conditionalconfidenceandposthocselectionnotindependenttemporalvalidation. Availabilitylag0scenario. No future shock success or publicdeploymentclaim.');

s=slide('Практические выводы из исследования',20,'Результаты относятся к изученному архиву 2024. Полные условия и ограничения — в отчёте.');
const benefits=[['Сезонный профиль задаёт сильный ориентир','На всех горизонтах ошибка в 1,3–2,3 раза ниже Prophet с профилем на архиве 2024.'],['Региональный остаток уменьшает очередь проверки','0,597 → 0,075 эпизода / 100 МО-месяцев. Орск: 16-е место из 2012, без тревоги.'],['Foundation-моделям помогает сезонное отношение','TimesFM h1: −45% к zero-shot, но выбранная модель остаётся точнее.'],['Рост следует проверять отдельно по категориям','Маркетплейсы: общая формула проигрывает на h3 / h6 / h12.']];
benefits.forEach(([title,body],i)=>{let y=158+i*106;txt(s,title,64,y,1152,38,28,true,C.green);txt(s,body,64,y+43,1152,52,23);});
txt(s,'Начать: README → критерий → доказательство → python3.12 run_review.py --mode verify',64,589,1152,29,20,false,C.muted);
notes(s,'Usefuloutputsareimplementedresearchartifactsandcachedviewer, notliveforecastservice. No new scientificresultorcontestscore. Verifychecksfilesonly,noMAEretraining. Fullarchive required; weightsnotincluded. Finalbenefitsslidefollowslimitationsasrequested.');

s=slide('Приложение: накопительный сигнал',21,'Апрель–декабрь 2024. Центр и порог по январю–марту; случаи уже были известны при разработке гипотезы.');
table(s,[['Метод','МО-месяцы','Тревожные месяцы','Доля, %'],...d.expansion.yoy.map(x=>[x.method==='cusum'?'CUSUM':'EWMA',num(x.monitoring_months),num(x.active_months),x.active_per100.toFixed(2)])],170,215,[350,245,320,237],27);
txt(s,'Сигнал: годовой рост МО относительно других МО региона\nСлишком высокая нагрузка для замены основного детектора',64,430,1152,93,28,true,C.green);
txt(s,'Для проверки отдельных МО готов локальный интерфейс\nРасходы, прогноз, интервалы, очередь тревог и источники',64,550,1152,60,27);
notes(s,'Source: yoy_shift_review/monitoring_burden.csv, protocol. logyoyleaveoneoutregionalmedian, JanMarcalibration99percentile, AprDec18201targets, nofuturethresholdinputsbutcasesknown. NoOrskAprJunalarm; realFARunknown dueunlabeledabsence. Dashboardofflinecachedresults only, notlivepredictionsorconfirmedfuturealerts.');

const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<21;i++){const slide=p.slides.items[i];const preview=await p.export({slide,format:'png',scale:1});await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await preview.arrayBuffer()));}
const output=path.join(root,`artifacts/final/task2-${Date.now()}.pptx`);
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:output,pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit", ...[2,5,6,7,8,9,10,13,14,18,21].flatMap(n=>["--require-native-table-slide",String(n)])],explicitTotalSlideCount:21,requiredNativeTableOwnerSlides:[2,5,6,7,8,9,10,13,14,18,21],requiredNativeChartOwnerSlides:[5,12,13,15],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,`validation-${Date.now()}.json`)});
await fs.copyFile(output,path.join(root,'artifacts/presentation.pptx'));console.log('Final deck:',output);
