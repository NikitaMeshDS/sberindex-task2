/* Read-only viewer. Forecasts, events and alarm states are saved research outputs. */
(async () => {
  "use strict";
  const d=await window.RESEARCH_DATA_READY, el=id=>document.getElementById(id), NS="http://www.w3.org/2000/svg";
  if(!d){document.querySelector("main").textContent="Данные не загружены. Запустите tools/build_research_dashboard.py из корня проекта.";return;}
  const cities=Object.values(d.municipalities), monthNames=["янв","фев","мар","апр","май","июн","июл","авг","сен","окт","ноя","дек"], rub=x=>x==null?"Нет факта":Math.round(x).toLocaleString("ru-RU"), pct=x=>x==null?"—":(x>0?"+":"")+x.toFixed(1)+"%";
  let selected=d.municipalities[1673]||cities[0], mapped=[], visible=[];
  const category=()=>el("category").value, horizon=()=>Number(el("horizon").value), month=()=>el("month").value, signal=()=>el("signal").value;
  const prediction=(c,m=month())=>c.forecasts[category()]?.[horizon()]?.[m], actual=(c,m=month())=>c.history[category()]?.[m];
  const error=c=>{let a=actual(c),p=prediction(c);return a>0&&p?(a-p[0])/a*100:null;};
  const svg=(tag,attrs={},text)=>{let n=document.createElementNS(NS,tag);for(let[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!=null)n.textContent=text;return n;};
  const nextMonth=m=>{const[y,z]=m.split("-").map(Number);return z===12?`${y+1}-01-01`:`${y}-${String(z+1).padStart(2,"0")}-01`;};
  function choose(id){let c=d.municipalities[id];if(!c)return;selected=c;renderMap();renderTable();renderDetail();}
  function setOptions(){
    let old=month(), available=new Set();for(const c of cities)for(const m of Object.keys(c.forecasts[category()]?.[horizon()]||{}))available.add(m);
    el("month").replaceChildren();for(const m of [...available].sort()){let o=document.createElement("option");o.value=m;o.textContent=`${monthNames[+m.slice(5)-1]} ${m.slice(0,4)}`;el("month").append(o);}
    if(available.has(old))el("month").value=old;else if(available.has("2024-05"))el("month").value="2024-05";
  }
  for(const c of d.categories){let o=document.createElement("option");o.value=c;o.textContent=c;el("category").append(o);}el("category").value="Все категории";
  function color(e){if(e==null)return"#c5cfc9";let t=Math.min(Math.abs(e)/25,1),v=e>=0?[185,98,32]:[47,109,155];return `rgb(${v.map(x=>Math.round(222*(1-t)+x*t)).join(",")})`;}
  function project(c){let lon=c.lon<0?c.lon+360:c.lon;return[36+(lon-19)/173*754,350-(c.lat-40)/42*320];}
  function renderMap(){
    const map=el("map");map.replaceChildren();mapped=[];
    const polygons=d.outline.type==='MultiPolygon'?d.outline.coordinates:[d.outline.coordinates];
    for(const rings of polygons){let line=rings.map(ring=>ring.map(([lon,lat],i)=>{let[x,y]=project({lon,lat});return(i?'L':'M')+x+','+y;}).join(' ')+' Z').join(' ');map.append(svg('path',{d:line,class:'country-outline'}));}
    for(let lon=30;lon<=180;lon+=30){let[x]=project({lon,lat:40});map.append(svg("line",{x1:x,x2:x,y1:22,y2:354,class:"map-grid"}),svg("text",{x,y:374,class:"map-label","text-anchor":"middle"},lon+"°"));}
    for(let lat=45;lat<=75;lat+=10){let[,y]=project({lon:19,lat});map.append(svg("line",{x1:30,x2:795,y1:y,y2:y,class:"map-grid"}),svg("text",{x:3,y:y+3,class:"map-label"},lat+"°"));}
    let data=visible.filter(c=>prediction(c)&&actual(c)!=null);for(const c of data){if(c.lat==null||c.lon==null)continue;let[x,y]=project(c);if(x<0||x>820||y<0||y>390)continue;let point=svg("circle",{cx:x,cy:y,r:c.id===selected.id?5.5:2.7,fill:color(error(c)),class:"point"+(c.id===selected.id?" selected":""),"data-id":c.id});point.append(svg("title",{},`${c.name}, ${c.region}\nОшибка ${pct(error(c))}`));point.addEventListener("click",()=>choose(c.id));point.addEventListener("pointermove",e=>{let t=el("tooltip");t.hidden=false;t.textContent=`${c.name}\n${c.region}\nОшибка ${pct(error(c))}`;t.style.left=Math.min(e.clientX+12,window.innerWidth-265)+"px";t.style.top=(e.clientY+12)+"px";});point.addEventListener("pointerleave",()=>el("tooltip").hidden=true);map.append(point);mapped.push(c);}
    let selectedPoint=map.querySelector(`[data-id="${selected.id}"]`);if(selectedPoint)map.append(selectedPoint);
    el("map-count").textContent=`${mapped.length.toLocaleString("ru-RU")} точек`;
    el("no-map").textContent=data.length?`${data.length} МО с фактом и прогнозом; ${data.length-mapped.length} без отображаемых координат.`:"В этом выборе нет сопоставимых фактов и прогнозов.";
  }
  function renderTable(){
    const body=el("cities");body.replaceChildren();const rows=visible.filter(c=>prediction(c)&&actual(c)!=null).sort((a,b)=>Math.abs(error(b)??0)-Math.abs(error(a)??0));
    for(const c of rows.slice(0,60)){let tr=document.createElement("tr");if(c.id===selected.id)tr.className="selected";let name=document.createElement("td"),btn=document.createElement("button"),region=document.createElement("span");btn.type="button";btn.textContent=c.name;btn.addEventListener("click",()=>choose(c.id));region.className="region-small";region.textContent=c.region;name.append(btn,region);tr.append(name);for(const value of [rub(actual(c)),rub(prediction(c)[0]),pct(error(c))]){let td=document.createElement("td");td.textContent=value;tr.append(td);}body.append(tr);}
    el("table-count").textContent=`Показаны ${Math.min(rows.length,60)} из ${rows.length} МО с наибольшей абсолютной относительной ошибкой. Для остальных используйте поиск.`;
  }
  function path(values,x,y){let s="",open=false;values.forEach((v,i)=>{if(v==null||!Number.isFinite(v)){open=false;return;}s+=(open?" L":" M")+x(i)+","+y(v);open=true;});return s;}
  function renderDetail(){
    const c=selected;el("city-name").textContent=c.name;el("city-region").textContent=`${c.region} · ID ${c.id}`;el("city-error").textContent=pct(error(c));
    const a=c.alarms[signal()]?.[month()];el("city-alert").textContent=!a?.observed?"Нет данных":a.episode?"Новый эпизод":a.active?"Продолжается":"Нет тревоги";
    const chart=el("forecast"), months=d.months;chart.replaceChildren();const facts=months.map(m=>actual(c,m)??null),forecasts=months.map(m=>prediction(c,m)?.[0]??null),lower=months.map(m=>prediction(c,m)?.[1]??null),upper=months.map(m=>prediction(c,m)?.[2]??null),all=[...facts,...forecasts,...lower,...upper].filter(v=>v!=null);
    if(!all.length){chart.append(svg("text",{x:40,y:150,class:"chart-axis"},"Нет данных для этого выбора"));}
    else{let lo=Math.min(...all),hi=Math.max(...all),pad=Math.max((hi-lo)*0.12,hi*0.025,1),min=Math.max(0,lo-pad),max=hi+pad,x=i=>62+i*562/(months.length-1), y=v=>266-(v-min)/(max-min)*226;
      for(let i=0;i<=4;i++){let v=min+(max-min)*i/4;chart.append(svg("line",{x1:62,x2:624,y1:y(v),y2:y(v),class:"chart-grid"}),svg("text",{x:55,y:y(v)+4,"text-anchor":"end",class:"chart-axis"},rub(v)));}
      for(let i=0;i<months.length;i++)chart.append(svg("text",{x:x(i),y:291,"text-anchor":"middle",class:"chart-axis"},monthNames[+months[i].slice(5)-1]));
      let segments=[],start=null;for(let i=0;i<=months.length;i++){if(i<months.length&&lower[i]!=null&&upper[i]!=null){if(start==null)start=i;}else if(start!=null){segments.push([start,i-1]);start=null;}}
      for(const[s,e]of segments){let pts=[];for(let i=s;i<=e;i++)pts.push(`${x(i)},${y(upper[i])}`);for(let i=e;i>=s;i--)pts.push(`${x(i)},${y(lower[i])}`);chart.append(svg("polygon",{points:pts.join(" "),class:"chart-band"}));}
      if(c.id===1673){let flood=months.indexOf('2024-04');if(flood>=0){chart.append(svg('line',{x1:x(flood),x2:x(flood),y1:35,y2:266,stroke:'#b96220','stroke-dasharray':'5 4'}),svg('text',{x:x(flood)+5,y:27,class:'chart-axis'},'Паводок: апрель'));}}
      let index=months.indexOf(month());if(index>=0)chart.append(svg("line",{x1:x(index),x2:x(index),y1:35,y2:266,stroke:"#b7cbbb","stroke-dasharray":"3 4"}));
      chart.append(svg("path",{d:path(facts,x,y),class:"chart-actual"}),svg("path",{d:path(forecasts,x,y),class:"chart-predicted"}));
      months.forEach((m,i)=>{if(c.alarms[signal()]?.[m]?.episode&&facts[i]!=null)chart.append(svg("circle",{cx:x(i),cy:y(facts[i]),r:5,class:"chart-alert"}));});
    }
    el("interval-note").textContent=lower.some(v=>v!=null)?"Полоса — иллюстративный интервал 80% из ошибок трёх прошлых месяцев; гарантия покрытия не подтверждена.":"Для этой категории и горизонта интервал не рассчитывался. Пропуски не соединены линией.";
    const p=prediction(c);el("selection-note").textContent=p?`Месяц факта: ${month()}. Последний известный месяц прогноза: ${p[3]}. Точки — последовательные прогнозы на ${horizon()} мес. Оранжевые точки — начало эпизода EWMA общего показателя.`:"Прогноз для выбранного месяца отсутствует.";
    const cutoff=nextMonth(month());el("news-date").textContent=`Публикации региона, доступные к ${cutoff} — условной дате получения факта.`;
    let news=d.news.filter(n=>n.region_code===c.region_code&&n.available_from<=cutoff&&(!n.municipality_ids.length||n.municipality_ids.includes(c.id))).sort((a,b)=>b.published.localeCompare(a.published)).slice(0,8),list=el("news");list.replaceChildren();
    for(const n of news){let li=document.createElement("li"),link=document.createElement("a"),date=document.createElement("span");link.href=n.url;link.target="_blank";link.rel="noopener noreferrer";link.textContent=n.title;date.textContent=`${n.published} · доступность ${n.available_from} · ${n.kind}`;li.append(link,date);list.append(li);}if(!news.length){let li=document.createElement("li");li.textContent="В сохранённом корпусе нет доступных публикаций для этого выбора. Покрытие неизвестно.";list.append(li);}
  }
  function renderQueue(){
    let rows=cities.filter(c=>c.alarms[signal()]?.[month()]?.episode).sort((a,b)=>{let aa=a.alarms[signal()][month()],bb=b.alarms[signal()][month()];return bb.score/bb.threshold-aa.score/aa.threshold;}),q=el("queue");q.replaceChildren();
    if(!rows.length){let p=document.createElement("p");p.className="empty";p.textContent="Новых эпизодов нет. Это не доказывает отсутствие изменений.";q.append(p);return;}
    let grid=document.createElement("div");grid.className="queue-list";for(const c of rows){let btn=document.createElement("button"),meta=document.createElement("span"),alarm=c.alarms[signal()][month()];btn.type="button";btn.className="queue-item";btn.textContent=c.name;meta.textContent=`${c.region} · сила ${ (alarm.score/alarm.threshold).toFixed(2)} × порог`;btn.append(meta);btn.addEventListener("click",()=>{el("search").value="";selected=c;render();document.querySelector(".detail").scrollIntoView({block:"center",behavior:"auto"});});grid.append(btn);}q.append(grid);
  }
  function render(){let query=el("search").value.toLocaleLowerCase("ru-RU").trim();visible=cities.filter(c=>!query||`${c.name} ${c.full_name} ${c.id}`.toLocaleLowerCase("ru-RU").includes(query));if(visible.length&&query&&!visible.includes(selected))selected=visible[0];renderMap();renderTable();renderDetail();renderQueue();if(query&&!visible.length)el("no-map").textContent="По запросу не найдено муниципалитетов. Очистите поиск или введите другое название/ID. Справа остаётся ранее выбранный МО.";}
  for(const id of ["category","horizon"])el(id).addEventListener("change",()=>{setOptions();render();});for(const id of ["month","signal"])el(id).addEventListener("change",render);el("search").addEventListener("input",render);
  document.querySelectorAll(".quick button").forEach(b=>b.addEventListener("click",()=>{el("search").value="";choose(b.dataset.id);render();}));
  el("data-coverage").textContent=`${cities.length} МО · ${d.categories.length} категорий · ${d.news.length} публикация(й)`;setOptions();render();el("load-status")?.remove();
})().catch(()=>{document.querySelector("main").textContent="Не удалось распаковать данные. Откройте проект в современном Chrome, Edge или Safari с поддержкой DecompressionStream.";});
