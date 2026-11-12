# Каталог литературы и источников данных для научных расчётов ArborScan

Дата проверки 11 ноября 2026 года. AS-10 AS-11 AS-13 и зависимости.

В каталоге 76 уникальных записей: научные публикации, наборы данных и официальная документация. Учитываются все восемь библиографических записей предоставленной статьи. Часть источников относится к прежнему обзору; это указано в глубине чтения. Аннотация, метаданные, выбранные разделы и полный текст различаются. Доступность ссылки не означает чтение всего документа.

## Библиография предоставленной статьи

### B1 Ребко Д.В. 2025-11

Анализ вероятности возникновения ветровалов на территории Республики Беларусь.

[Первоисточник](https://doi.org/10.33408/2519-237X.2025.9-4.551). DOI 10.33408/2519-237X.2025.9-4.551.

**Глубина чтения:** full text, all 11 PDF pages; extraction, not complete visual formula QA.

**Значение:** Mamdani prototype, illustrative risk outputs and limited validation; species beta values use kg/(m*s), distinct from total beta kg/s..

**Ограничение:** No reproducible risk calibration supplied by this article; current model artefacts not established by publication..

### B2 Gardiner B., Berry P., Moulia B. 2016-04; online 2016-01-27

Review: Wind impacts on plant growth, mechanics and damage.

[Первоисточник](https://www.sciencedirect.com/science/article/pii/S0168945216300061). DOI 10.1016/j.plantsci.2016.01.006.

**Глубина чтения:** abstract and publisher section previews; not full 25-page review.

**Значение:** Reconfiguration, dynamic loading and acclimation matter; tree mechanics varies with environment..

**Ограничение:** Does not supply an experimentally calibrated photo-to-beta regressor..

### B3 Borisevich S.A., Kamluk A.N., Rebko D.V. 2022-05; publisher preview states online 2022-02-11

Drag relationships for full-grown Scots pine trees.

[Первоисточник](https://cdnsciencepub.com/doi/10.1139/cjfr-2021-0043). DOI 10.1139/cjfr-2021-0043.

**Глубина чтения:** publisher abstract and first-page publisher preview (882); Crossref deposited publisher metadata. Prior project proof analysis read as historical evidence, not reread original proof here..

**Значение:** Five pine free-fall experiments; constant linear scaling parameter 25.5-90 kg/s in abstract. Final title is full-grown, not full-scale..

**Ограничение:** Final full text not obtained. No >100-tree training set or aspen photo regression verified..

### B4 Borisevich S.A., Vikhrenko V.S. 2018-06-15

Evaluation of the drag coefficients of tree crowns by numerical modeling of their free fall.

[Первоисточник](https://www.sciencedirect.com/science/article/abs/pii/S0168192318301035). DOI 10.1016/j.agrformet.2018.03.020.

**Глубина чтения:** abstract, introduction and publisher section snippets (methods/results/discussion/conclusion), not full seven-page paper.

**Значение:** Five symmetrical forest pines 23-27m near Minsk; free fall, linear drag, triangular distributed profile; intensity gamma_max has kg/(m*s), not automatically total beta..

**Ограничение:** No raw trajectories/Maple reproduction package obtained; complete integration method not recovered from previews..

### B5 Борисевич С.А., Камлюк А.Н., Ребко Д.В. 2013

Определение сопротивления движению кроны дерева.

[Первоисточник](https://elibrary.ru/TKARSZ). DOI не установлен.

**Глубина чтения:** bibliographic entry in supplied manuscript and in full primary B1 only; exact original not obtained.

**Значение:** Exact reference has been accounted for, not silently replaced by a different 2013 article..

**Ограничение:** No independent content-based conclusions attributed to this inaccessible original; DOI not established..

### B6 Friedman J.H. 2001-10-01

Greedy function approximation: A gradient boosting machine.

[Первоисточник](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full). DOI 10.1214/aos/1013203451.

**Глубина чтения:** 2001 publisher metadata/abstract; selected text sections 1-3 and5 (regularization) of author 1999-02-24 preprint. Scanned final PDF obtained through browser but body not read visually; not full final article..

**Значение:** Stagewise gradient boosting and shrinkage/iteration regularization are methodological basis, not evidence of any ArborScan trained model..

**Ограничение:** A citation to an algorithm does not establish implementation, training data, model weights or external validation..

### B7 Lin L.I. 1989-03

A concordance correlation coefficient to evaluate reproducibility.

[Первоисточник](https://pubmed.ncbi.nlm.nih.gov/2720055/). DOI 10.2307/2532051.

**Глубина чтения:** NLM-indexed abstract and publisher-deposited Crossref metadata; not full 14-page paper.

**Значение:** Agreement near the identity line differs from association. Table CCC recalculated independently..

**Ограничение:** CCC alone is not a validated safety decision or evidence of causal beta identifiability..

### B8 Bland J.M., Altman D.G. 1986-02

Statistical methods for assessing agreement between two methods of clinical measurement.

[Первоисточник](https://www-users.york.ac.uk/~mb55/meas/ba.pdf). DOI 10.1016/S0140-6736(86)90837-8.

**Глубина чтения:** full 9-page author-hosted reprint, text; uncertainty and repeatability sections included.

**Значение:** Bias and limits of agreement must be judged against prespecified use limits; small samples give imprecise limits; correlation is not agreement..

**Ограничение:** Does not justify universal acceptable error bands for beta or tree failure..

## Бета и статистика

### A1 Определение сопротивления кроны дерева при его движении: экспериментальные результаты и численные расчеты

Камлюк А.Н., Борисевич С.А., Ребко Д.В.. 2013.

[Первоисточник](https://agz.edu.kz/public/uploads/raznoe/Vestnik_11.pdf).

**Глубина проверки:** full article text, PDF pages12-18 (indices11-17) within104-page issue; not whole issue.

**Применение:** Cut-edge rotation axis without leaving stump, triangular beta_max; initial angular speed estimated from early motion then drag from later motion.

**Граница:** Different publication from B5. Ambiguous beta_max/total-beta relation and discretization require explicit definitions; not >100 tree regression evidence.

### A2 Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood

Raue A. et al.. 2009-08-01; online2009-06-08.

[Первоисточник](https://academic.oup.com/bioinformatics/article/25/15/1923/213246). DOI 10.1093/bioinformatics/btp358.

**Глубина проверки:** abstract and indexed publisher method excerpts; two author/institutional PDF fetches timed out; not full.

**Применение:** Profile likelihood detects structural/practical non-identifiability and informs experiment design.

**Граница:** Method adaptation to tree dynamics is a proposed design, not a result in the paper.

### A3 Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure

Roberts D.R. et al.. 2017 issue40(8),913-929; online2016-12-08.

[Первоисточник](https://nsojournals.onlinelibrary.wiley.com/doi/full/10.1111/ecog.02881). DOI 10.1111/ecog.02881.

**Глубина проверки:** abstract and selected full-text sections on structured data and cross-validation objectives; not all simulations/references.

**Применение:** Blocked validation should match deployment objective and dependence structure; tree/site/campaign grouped splits proposed.

**Граница:** No universally optimal block size or performance guarantee for Belarus data.

### A4 Conformalized Quantile Regression

Romano Y., Patterson E., Candès E.J.. 2019 NeurIPS32; preprint2019-05-08.

[Первоисточник](https://proceedings.neurips.cc/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf). DOI 10.48550/arXiv.1905.03222.

**Глубина проверки:** selected full-text pages1-7 and9: methods1-5, Algorithm1, Theorem1/2 and conclusion; not supplementary proofs or all experiment details.

**Применение:** Separate training/calibration; conformity score correction; finite-sample marginal coverage under exchangeability, not guaranteed per individual or OOD urban case.

**Граница:** Prediction intervals must be calibrated on the same target/measurement regime; reference-label error is not automatically removed.

### A5 Natural sway frequencies and damping ratios of trees: concepts, review and synthesis of previous studies

Moore J.R., Maguire D.A.. 2004-03.

[Первоисточник](https://napier-repository.worktribe.com/output/259422/natural-sway-frequencies-and-damping-ratios-of-trees-concepts-review-and-synthesis-of-previous-studies). DOI 10.1007/s00468-003-0295-6.

**Глубина проверки:** author-institutional abstract/metadata; not full review.

**Применение:** Damping combines internal, aerodynamic and crown-contact processes; aerodynamic damping can depend nonlinearly on speed.

**Граница:** A fitted sway damping ratio is not directly the total free-fall crown beta.

### A6 N11 An architectural understanding of natural sway frequencies in trees

Jackson T. et al.. 2019-06-05.

[Первоисточник](https://pmc.ncbi.nlm.nih.gov/articles/PMC6597771/). DOI 10.1098/rsif.2019.0116.

**Глубина проверки:** selected full-text abstract, Introduction1.1/1.2 and Methods2.1/2.2; not all finite-element sensitivity analyses.

**Применение:** Architecture/leaf state matters; field pull-release fitted strain decay. Detailed QSM architecture is not equivalent to a single photograph.

**Граница:** Does not validate the article beta regressor or prescribe a unique damping-transfer map.

Повторные ссылки на тот же DOI объединены; подробная разница чтения сохранена в JSON.

### A7 In situ drag coefficient measurements for rooftop trees

Koizumi A., Shimizu M., Sasaki Y., Hirai T.. 2016-05-18; issue2016-08.

[Первоисточник](https://link.springer.com/article/10.1007/s10086-016-1556-5). DOI 10.1007/s10086-016-1556-5.

**Глубина проверки:** publisher abstract and metadata; subscription full text not accessed.

**Применение:** Wind/deflection measurement in built setting provides a relevant independent standing-tree calibration direction.

**Граница:** Only two reported rooftop specimens, no universal urban-tree beta or transfer factor.

## Механика дерева

### O01 Comparison of two models for predicting the critical wind speeds required to damage coniferous trees

Gardiner, Peltola, Kellomaki. 2000.

[Первоисточник](https://doi.org/10.1016/S0304-3800(00)00220-9). DOI 10.1016/S0304-3800(00)00220-9.

**Глубина проверки:** Prior report: selected full author text; not reread here.

**Применение:** forest stands; GALES/HWIND.

### O02 Wind loading of trees: influence of tree size and competition

Hale et al.. 2012.

[Первоисточник](https://doi.org/10.1007/s10342-010-0448-2). DOI 10.1007/s10342-010-0448-2.

**Глубина проверки:** Prior report: methods, sample and relationships from full author text; not reread here.

**Применение:** standing trees within stands; TMC.

### O03 Comparison and validation of three versions of a forest wind risk model

Hale et al.. 2015.

[Первоисточник](https://doi.org/10.1016/j.envsoft.2015.01.016). DOI 10.1016/j.envsoft.2015.01.016.

**Глубина проверки:** Prior report: full article, selected formula/result pages; supplement not obtained; not reread here.

**Применение:** forest stand event validation; TMC and roughness.

### O04 Estimating stem and root-anchorage flexibility in trees

Neild and Wood. 1999.

[Первоисточник](https://doi.org/10.1093/treephys/19.3.141). DOI 10.1093/treephys/19.3.141.

**Глубина проверки:** Prior report: full author text selected mathematics and validation; not reread here.

**Применение:** standing tree bending and root rotation.

### O05 Anchorage of coniferous trees in relation to species, soil type, and rooting depth

Nicoll et al.. 2006.

[Первоисточник](https://doi.org/10.1139/X06-072). DOI 10.1139/X06-072.

**Глубина проверки:** Prior report: full author text methods/results/tables/discussion; not reread here.

**Применение:** forest pulling tests; root-rot trees excluded from relevant regression.

### O06 Modelling the anchorage of shallow-rooted trees

Achim and Nicoll. 2009.

[Первоисточник](https://doi.org/10.1093/forestry/cpp004). DOI 10.1093/forestry/cpp004.

**Глубина проверки:** Prior report: full author text methods/equations/tables/discussion; not reread here.

**Применение:** 148 Sitka spruce; shallow root-plate model.

### O07 Variance-based sensitivity analysis of a wind risk model - Model behaviour and lessons for forest modelling

Locatelli et al.. 2017.

[Первоисточник](https://doi.org/10.1016/j.envsoft.2016.10.010). DOI 10.1016/j.envsoft.2016.10.010.

**Глубина проверки:** Prior report: selected methods/inputs/discussion in author manuscript; not reread here.

**Применение:** numerical sensitivity of forest model; correlated inputs.

### O08 Presence of Root Rot Reduces Stability of Norway Spruce (Picea abies): Results of Static Pulling Tests in Latvia

Krisans et al.. 2020.

[Первоисточник](https://doi.org/10.3390/f11040416). DOI 10.3390/f11040416.

**Глубина проверки:** Prior report: full HTML methods/results/tables1-3/conclusions; not reread here.

**Применение:** 77 spruce, static destructive pulling; root rot.

### O09 Predicting the risk of individual tree fall along powerlines in Norway with a mechanistic wind risk model and machine learning

Merlin, Gardiner and Solberg. 2026.

[Первоисточник](https://doi.org/10.5194/nhess-26-2461-2026). DOI 10.5194/nhess-26-2461-2026.

**Глубина проверки:** Prior report: full HTML methods/results/discussion/appendices/data-code statement; not reread here.

**Применение:** individual trees at forest powerline corridors; no major storm during winter.

### N01 Wind load estimation on an open-grown European oak tree

Angelou, Dellwik and Mann. 2019.

[Первоисточник](https://doi.org/10.1093/forestry/cpz026). DOI 10.1093/forestry/cpz026.

**Глубина проверки:** author-institution repository abstract and bibliographic record.

**Применение:** living rooted open-grown oak, leaf-on and leaf-off.

**Граница:** Full article not read here; one oak campaign overlaps later DTU papers.

### N02 Drag coefficient and frontal area of a solitary mature tree

Bekkers, Angelou and Dellwik. 2022.

[Первоисточник](https://doi.org/10.1016/j.jweia.2021.104854). DOI 10.1016/j.jweia.2021.104854.

**Глубина проверки:** selected full PDF text: sections1-3,6.3-7, Eq2; no visual formula audit.

**Применение:** living rooted solitary Quercus robur; 6.5m tall,8.5m crown; controlled direction sector at coastal site.

**Граница:** One stiff oak and narrow exposure; not an urban-canyon or extreme-storm validation; same research campaign as N01/N03.

### N03 Mean and maximum two dimensional wind force on an open-grown tree

Angelou, Gardiner and Dellwik. 2025.

[Первоисточник](https://doi.org/10.1016/j.jweia.2024.105966). DOI 10.1016/j.jweia.2024.105966.

**Глубина проверки:** selected full PDF text: calibration/methods, discussion, AppendixB, data statement.

**Применение:** living rooted rural open-grown oak; summer and winter.

**Граница:** Moderate measured wind range, one tree/campaign; raw data on request; paper beta is power-law premultiplier, not AS-10 linear beta.

### N04 Wind tunnel measurements of crown streamlining and drag relationships for several hardwood species

Vollsinger, Mitchell, Byrne, Novak and Rudnicki. 2005.

[Первоисточник](https://doi.org/10.1139/X05-051). DOI 10.1139/X05-051.

**Глубина проверки:** author-uploaded full HTML text: abstract and methods, pp1239-1240; not all result tables.

**Применение:** freshly cut juvenile crowns, recut to1.9m, kept moist; fixed wind-tunnel fixture.

**Граница:** No roots; limited size and 4-20m/s near-uniform tunnel flow; not a validation of mature standing-tree dynamics; webpage year2011 conflicts with actual2005 issue.

### N05 Analysis of the probability of failure for open-grown trees during wind storms

Ciftci, Arwade, Kane and Brena. 2014.

[Первоисточник](https://doi.org/10.1016/j.probengmech.2014.04.002). DOI 10.1016/j.probengmech.2014.04.002.

**Глубина проверки:** publisher abstract, introduction and selected methods/results preview; full text not read.

**Применение:** two modelled sugar maples with measured geometry; numerical storms; fixed base.

**Граница:** Failure is stem yield at1.4m, not uprooting; assumed damping5% winter/15% summer; no Belarus calibration; not annual probability or target risk.

### N06 Impact of Trenching on Root Loss and Tree Stability

Miesbauer, Koeser and Kane. 2025.

[Первоисточник](https://doi.org/10.48044/jauf.2024.026). DOI 10.48044/jauf.2024.026.

**Глубина проверки:** author-uploaded full HTML text: methods pp95-96, results/discussion pp98-100.

**Применение:** 30 established rooted Acer rubrum, pre/post trenching, pull-to1degree, then root excavation.

**Граница:** Young trees DBH9.9-17cm, one site, acute response; tilt/stress proxy is not direct ultimate uprooting load or long-term failure probability.

### N07 The Effects of Pruning on Drag and Bending Moment of Shade Trees

Pavlis, Kane, Harris and Seiler. 2008.

[Первоисточник](https://doi.org/10.48044/jauf.2008.028). DOI 10.48044/jauf.2008.028.

**Глубина проверки:** full PDF selected methods pp207-209 and discussion pp212-213.

**Применение:** trees cut above root flare, mounted on pickup truck; three shade-tree species.

**Граница:** No root coupling; roughly4-5m trees, not mature street-tree failure outcomes; pruning alters physiology and architecture too.

### N08 Experimental Test of Non-Destructive Methods to Assess the Anchorage of Trees

Detter, Rust and Krisans. 2023.

[Первоисточник](https://doi.org/10.3390/f14030533). DOI 10.3390/f14030533.

**Глубина проверки:** full HTML selected methods,results,discussion; Eq1 and definition of kr; not a visual PDF audit.

**Применение:** more than280 rooted trees on urban/forest sites; crowns removed before nondestructive then destructive pull.

**Граница:** Their kr is a moment at fixed tilt, not Nm/rad spring stiffness; size/species/extrapolation error; modified tree not complete natural-wind validation; six nonlinear fits excluded.

### N09 Root Pruning and Stability of Young Willow Oak

Smiley. 2008.

[Первоисточник](https://doi.org/10.48044/jauf.2008.016). DOI 10.48044/jauf.2008.016.

**Глубина проверки:** full PDF selected methods and results, pp123-126.

**Применение:** young rooted Quercus phellos; repeated1degree pulls; subset pulled to failure.

**Граница:** One young species/site; no universal safety distance or percentage-capacity multiplier for all Belarus trees.

### N10 Loss in moment capacity of tree stems induced by decay

Ciftci, Kane, Brena and Arwade. 2014.

[Первоисточник](https://doi.org/10.1007/s00468-013-0968-8). DOI 10.1007/s00468-013-0968-8.

**Глубина проверки:** publisher abstract/preview; author PDF fetch timed out.

**Применение:** cross-section mechanics plus selected oak winching experiments.

**Граница:** Exact equations and all validation data not read; not justification for a universal hollow ratio or automatic capacity from one photo.

### N12 The motion of trees in the wind: a data synthesis

Jackson et al.. 2021.

[Первоисточник](https://doi.org/10.5194/bg-18-4059-2021). DOI 10.5194/bg-18-4059-2021.

**Глубина проверки:** full HTML selected methods,discussion,conclusions,data/code statements.

**Применение:** 243 living trees from20studies,4-20Hz sensor data; broadleaf/conifer and open/forest.

**Граница:** Many datasets lack collocated high-frequency wind; not a ready aerodynamic beta or failure validation dataset; input studies overlap some other papers.

### N13 Natural sway frequencies and damping ratios of trees: influence of crown structure

Moore and Maguire. 2005.

[Первоисточник](https://doi.org/10.1007/s00468-004-0387-y). DOI 10.1007/s00468-004-0387-y.

**Глубина проверки:** publisher abstract/preview only.

**Применение:** nine rooted plantation Douglas-fir, progressive crown removal.

**Граница:** Full methods not read; small conifer sample; cannot transfer damping constants to urban broadleaves.

### N14 Crown structure and wood properties: Influence on tree sway and response to high winds

Sellier and Fourcaud. 2009.

[Первоисточник](https://doi.org/10.3732/ajb.0800226). DOI 10.3732/ajb.0800226.

**Глубина проверки:** full HTML selected methods,load scenarios,results/discussion.

**Применение:** virtual35year maritime pine;3D beams calibrated to forest architecture.

**Граница:** Numerical response study; root/stem failure resistance not modelled; a reported dynamic amplification is not a universal multiplier.

### N15 A Study of Branch Dynamics on an Open-Grown Tree

James. 2014.

[Первоисточник](https://doi.org/10.48044/jauf.2014.014). DOI 10.48044/jauf.2014.014.

**Глубина проверки:** full PDF selected methods and discussion, pp125-130.

**Применение:** one rooted19.7m silver maple with4codominant branches; successive pruning and pull-release.

**Граница:** Single tree, artificial excitation, not a universal pruning rule or capacity measurement.

### N16 Effects of crown structure on the sway characteristics of large decurrent trees

Kane, Modarres-Sadeghi, James and Reiland. 2014.

[Первоисточник](https://doi.org/10.1007/s00468-013-0938-1). DOI 10.1007/s00468-013-0938-1.

**Глубина проверки:** publisher abstract/preview only.

**Применение:** eight large open-grown sugar maples.

**Граница:** Eight trees in one region; empirical predictors are not validated for all urban species/conditions.

## Данные и сервисы

### era5 C3S ERA5 hourly single levels

[Первоисточник](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview).

**Глубина проверки:** Dataset overview/data description/licence label; no CDS download.

**Применение:** core climate source candidate; local/station validation required.

**Граница:** individual-tree canopy wind, urban building wakes, root strength, beta or validated annual failure probability.

### era5-land C3S ERA5-Land hourly

[Первоисточник](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview).

**Глубина проверки:** Overview, forcing, grid, depth/time, licence label; no download.

**Применение:** optional environment/state covariate, not substitute for wind validation.

**Граница:** water table at tree, rootable depth, tree-specific anchorage or more accurate local wind by resolution alone.

### openmeteo-archive Open-Meteo Historical Weather API

[Первоисточник](https://open-meteo.com/en/docs/historical-weather-api).

**Глубина проверки:** Model options/variables/time coverage and consistency recommendation.

**Применение:** quick research adapter/fallback; pin models=era5.

**Граница:** measured tree wind or validated AEP.

### openmeteo-terms Open-Meteo Terms

[Первоисточник](https://open-meteo.com/en/terms).

**Глубина проверки:** Non-commercial limits, data licence and commercial endpoint distinction.

### openweather OpenWeather Current Weather API / guide

[Первоисточник](https://openweathermap.org/current).

**Глубина проверки:** Current variable/key contract; no authenticated request.

**Применение:** retain current weather display only; never use one snapshot for climate AEP.

**Граница:** long-term wind distribution, return period, tree-scale gust field or beta.

### openweather-license OpenWeather guide licensing

[Первоисточник](https://openweathermap.org/guide).

**Глубина проверки:** Key/subscription workflow and current licensing summary.

### noaa-isd NOAA NCEI Global Hourly / ISD

[Первоисточник](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database).

**Глубина проверки:** Coverage, variables, QC, update/access descriptions and one station/day sample.

**Применение:** essential independent station comparison where temporal coverage is sufficient.

**Граница:** wind at tree in an urban courtyard or forest understory; seven reports are not24hourlymeans.

### noaa-api NCEI Access Data Service API documentation

[Первоисточник](https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation).

**Глубина проверки:** Query/dataTypes/location/name response contract.

### noaa-wind-format NOAA ISD record-format specification

[Первоисточник](https://www.ncei.noaa.gov/nerms/api/document/5b2ced2e766e4f7091d66ab228a58813/download).

**Глубина проверки:** Wind speed scaling, observation-type codes and QC definitions only.

### belhydromet Белгидромет / Pogoda.by official portals

[Первоисточник](https://belgidromet.by/).

**Глубина проверки:** Direct official-page opening failed; pogoda.by yielded no readable lines; no documented public hourly API established.

**Применение:** desired national validation partner/input, not an implemented dependency.

**Граница:** automatic full national availability or permission to redistribute.

### gwa4 DTU/World Bank Global Wind Atlas4 dataset

[Первоисточник](https://data.dtu.dk/articles/dataset/Global_Wind_Atlas_4/28955267). DOI 10.11583/DTU.28955267.v1.

**Глубина проверки:** Full repository description/file catalogue/licence through Figshare API; no raster downloaded.

**Применение:** secondary wind-climate prior and exposure cross-check, not AEP calibration by itself.

**Граница:** annual extreme/failure probability, urban wakes, individual tree effective wind or beta.

### gwa4-method Global Wind Atlas official introduction

[Первоисточник](https://globalwindatlas.info/en/about/introduction).

**Глубина проверки:** 4.0 model changes and histogram generalisation; website direct view JS shell.

### gwa4-validation DTU Global Wind Atlas4 Validation report

[Первоисточник](https://backend.orbit.dtu.dk/ws/portalfiles/portal/400877326/DTU_Wind_Energy_Report-E-0256.pdf).

**Глубина проверки:** Method/2008-2017 reference period and5heights sections; not full report read.

### soilgrids ISRIC SoilGrids layers

[Первоисточник](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html).

**Глубина проверки:** Layer names/scaling/depths/quantiles/mask/update descriptions.

**Применение:** background context/prior only, never automatic measured rooting or A-D mechanical class.

**Граница:** rootable depth, actual groundwater/compaction/root damage, anchorage strength or beta.

### soilgrids-access ISRIC SoilGrids access

[Первоисточник](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_02.html).

**Глубина проверки:** Licence/access methods and REST paused notice.

### soilgrids-wcs ISRIC WCS protocol

[Первоисточник](https://docs.isric.org/globaldata/soilgrids/wcs.html).

**Глубина проверки:** Capabilities/DescribeCoverage/GetCoverage, CRS/subset parameter contract.

### soilgrids-local-limit ISRIC SoilGrids FAQ

[Первоисточник](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_04.html).

**Глубина проверки:** No single year and caution against local/farm interpretation.

### copdem Copernicus DEM official collection

[Первоисточник](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).

**Глубина проверки:** DSM definition/acquisition/resolution/free licence plus public tile HEAD only.

**Применение:** regional topography/exposure context; field/more preciseDTM needed for critical local cases.

**Граница:** bare-earth microtopography, exact tree height by DSM subtraction, root geometry or urban flow.

### worldcover ESA WorldCover data access

[Первоисточник](https://esa-worldcover.org/en/data-access).

**Глубина проверки:** Product/tile/access metadata; no pixel value sampled.

**Применение:** landscape/edge candidate detection, user/field confirmation for model inputs.

**Граница:** species, stem spacing, tree height, exact gaps or recent felling.

### worldcover-manual ESA WorldCover2021 Product User ManualV2

[Первоисточник](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf).

**Глубина проверки:** Product versions/accuracy/comparability introduction only; not complete manual read.

### osm OpenStreetMap copyright

[Первоисточник](https://www.openstreetmap.org/copyright?locale=en-GB).

**Глубина проверки:** Data licence/attribution.

**Применение:** city surroundings/shelter context; never directCFD/structural safety evidence.

**Граница:** certified building heights, complete tree inventory or local turbulence.

### overpass Overpass official community API documentation

[Первоисточник](https://wiki.openstreetmap.org/wiki/Overpass_API).

**Глубина проверки:** Read-only contract/public-instance usage/overload limits and extracts recommendation.

### ghsl EC-JRC GHS-BUILT-H R2023A

[Первоисточник](https://human-settlement.emergency.copernicus.eu/ghs_buH2023.php).

**Глубина проверки:** Dataset identity/source epoch;100m technical specification via EC/JRC source extract.

**Применение:** optional fallback urban context, lower priority than mapped/measured buildings.

**Граница:** courtyard wakes, wall geometry or current building-by-building obstruction.

### ghsl-catalog EC-JRC GHSL datasets and reuse notice

[Первоисточник](https://human-settlement.emergency.copernicus.eu/datasets.php).

**Глубина проверки:** Public/free availability and source acknowledgment statement.

### gedi GEDI instrument specifications

[Первоисточник](https://gedi.umd.edu/instrument/specifications/).

**Глубина проверки:** Latitude/footprint/sampling geometry.

**Применение:** not a Belarus primary field-reference dataset.

**Граница:** GEDI measured height inMinsk or wall-to-wallBelarus canopy.

### glad2019 UMD GLAD Global Forest Canopy Height2019

[Первоисточник](https://www.glad.geog.umd.edu/dataset/gedi/).

**Глубина проверки:** Explicit data-provided-only-withinGEDI-range notice and map metadata.

**Применение:** exclude from initial national pipeline.

**Граница:** Minsk tree height or current per-tree geometry.

### eth-canopy ETH global canopy height2020

[Первоисточник](https://langnico.github.io/globalcanopyheight/).

**Глубина проверки:** Method/global coverage/height and uncertainty layers/licence.

**Применение:** regionalcanopy prior/range check with uncertainty, no silent substitution of measuredgeometry.

**Граница:** height/diameter of selected individual tree or top100dominantstems/ha bydefinition.

### gwdd2 Global Wood Density Databasev2.2

[Первоисточник](https://zenodo.org/records/20815517).

**Глубина проверки:** Full recordmetadata/licence,1.71MBbinomialaggregate and3.5KBcolumnmetadata; no73-85MBrawfiles.

**Применение:** curated versioned speciesprior with densitybasis retained; neverrename togreen-density.

**Граница:** freshlivingmassdensity,E,MOR,Creg,beta or condition/decay ofone tree.

### wood-handbook USDA FPL Wood Handbook2021 Chapter5

[Первоисточник](https://research.fs.usda.gov/treesearch/62244).

**Глубина проверки:** Variability/moisture/knots/slope-ofgrain scope; full46-pagechapter not read for this task.

**Применение:** referenceboundary; not automatically validated profile.

**Граница:** measuredE/MORofselectedtree or rootplatefailure.

### repola2009 Repola2009 pine/spruce componentbiomass

[Первоисточник](https://silvafennica.fi/article/184).

**Глубина проверки:** Scope/sample/species/components/inputdescription; no fullcoefficienttranscription.

**Применение:** candidate regional forestallometry, explicit domain and uncertainty.

**Граница:** freshmassdistribution,inertia,rootstrength,E,beta or validityforurbansolitarytrees.

### zianis2005 European biomass/stem-volume equation catalogue

[Первоисточник](https://silvafennica.fi/show/mono/mono4).

**Глубина проверки:** Equation catalogue scope, species/country/tree-size metadata; not all837equations reviewed.

**Применение:** source-discovery only, not defaultformula selector.

**Граница:** nationalcalibrationorfieldmeasurement.

### era5-temporal-semantics ECMWF ERA5 terminology current version

[Первоисточник](https://confluence.ecmwf.int/spaces/CKB/pages/85402030/ERA5%2Bterminology%2Banalysis%2Band%2Bforecast%2Btime%2Band%2Bsteps%2Binstantaneous%2Band%2Baccumulated%2Band%2Bmean%2Brates%2Band%2Bmin%2Bmax%2Bparameters).

**Глубина проверки:** Analysis/forecast and instantaneous/accumulated/mean/minmax definitions; followed current-version link from historical pageId111151514.

### era5-data-docs ECMWF ERA5 data documentation current version

[Первоисточник](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation).

**Глубина проверки:** Data organisation, analysis-on-CDS and temporal frequency; not entire64min-document read; historical pageId212448915 replaced by current-version link.

## Интеграция

### T01 GroupKFold

[Первоисточник](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).

**Глубина проверки:** Current official API documentation; grouped splitting contract.

**Применение:** Non-overlapping groups for tree-level cross-validation.

### T02 Common pitfalls and data leakage

[Первоисточник](https://scikit-learn.org/stable/common_pitfalls.html).

**Глубина проверки:** Official documentation, data leakage/preprocessing sections.

**Применение:** Fit transformations on training data; pipeline inside validation.

### T03 Supabase Row Level Security

[Первоисточник](https://supabase.com/docs/guides/database/postgres/row-level-security).

**Глубина проверки:** Official documentation, RLS and bypass/keys sections; no live database changed.

**Применение:** Owner isolation, server keys, RLS and grants are distinct.

### T04 Tables not exposed to Data and GraphQL API automatically

[Первоисточник](https://supabase.com/changelog/45329-breaking-change-tables-not-exposed-to-data-and-graphql-api-automatically).

**Глубина проверки:** Official changelog and relevant summary; self-hosted installed version not audited.

**Применение:** Verify grants for new tables separately from RLS and migrations.

## Проверяемость

Машиночитаемый каталог source-catalog.json хранит исходные поля каждого обзора, aliases, версии, лицензии, ограничения доступа и глубину. API-пробы отделены от документации в agent-data/api-checks.json. Полные чужие тексты и персональные контакты не включены в репозиторный каталог.
