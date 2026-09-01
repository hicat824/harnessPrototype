# Company Engine Code Skill

## Purpose

用于统一解释公司接口中的流程名称、`engineCode` / `flow_id` 等流程编码，并指导 Agent 自动补全输入参数和筛选接口返回结果。

## Core rules

1. 本 Skill 维护的是公司统一的流程名称与 `engineCode` 映射关系。
2. **只有当当前准备调用的 MCP Tool 参数定义中明确包含 `engine_code`、`engineCode`，或接口文档明确说明某参数使用同一套 engineCode 编码体系时，Agent 才需要根据用户语义自动补充对应的流程编码。**
3. 如果当前 Tool 不需要 engineCode 类参数，则即使用户提到了“入库”“标准化”等流程名称，也**不得额外生成、附加或传递 engineCode 参数**。
4. 当 Tool 确实需要 engineCode 类参数时：
   - 用户已经明确表达流程名称：根据本 Skill 映射自动转换并补参；
   - 当前上下文已经可靠确定目标流程：直接复用对应 engineCode；
   - 无法唯一确定：不得猜测，应询问用户或返回候选项。
5. `engineCode`、`engine_code`、`flow_id` 等字段只有在接口文档明确说明它们使用同一套流程编码体系时，才可以按本 Skill 的映射解释；不能仅凭字段名自行假设。
6. 对于接口**输出结果**，如果返回 `list`，且元素包含 `engineCode` 字段，并且当前用户请求已经明确目标流程，则应优先选择 `obj["engineCode"] == target_engine_code` 的对象。
7. 输出筛选必须优先使用 `engineCode` 精确匹配，不应仅依据数组顺序、中文描述、相似名称或其他不稳定字段判断。
8. 如果列表中没有目标 `engineCode`，明确说明未找到对应流程结果；如果存在多个相同 `engineCode`，不要擅自任选一个，应结合其他业务字段继续判断或向用户说明。
9. 在同一次工具调用链中，一旦目标流程已经可靠确定，可在后续**确实需要 engineCode 的 Tool 调用或结果筛选中**复用该流程编码；不需要 engineCode 的 Tool 不应因此增加额外参数。

### Input parameter example

假设 Tool 定义为：

```text
query_task_file(task_id, engine_code)
```

用户：`查询入库任务12345对应的点云文件路径`

如果“入库”在当前业务语义中对应 `import-std`，则 Agent 应自动形成：

```text
task_id = "12345"
engine_code = "import-std"
```

但如果 Tool 定义为：

```text
query_task_file(task_id)
```

则只应传：

```text
task_id = "12345"
```

即使用户提到了“入库”，也不能额外构造 `engine_code`。

### Output filtering example

如果当前请求已经确定目标流程为“入库” -> `import-std`，某接口返回：

```json
[{"engineCode":"import-slam","path":"..."},{"engineCode":"import-std","path":"/data/xxx.pcd"}]
```

则选择：

```text
engineCode == "import-std"
```

对应的对象，再从该对象中提取用户需要的 `path`。

> 注意：源表中标准化对应的实际编码是 `import-std`（连字符），不是 `import_std`（下划线）。Agent 必须使用映射表中的原始值。

## Special rules

- 对 `datum_import` 组中的数据前处理流程，源表明确标注：`engineCode` 中的 `dataSource` 需要使用小写。涉及这类动态编码或参数时，应保持 `dataSource` 小写。
- 中文名称为空的 engineCode 仍是合法编码，但 Agent 不应自行创造其业务中文含义。
- 本 Skill 不意味着所有公司接口都需要 engineCode。是否需要传递 engineCode，必须以当前 MCP Tool 的参数 schema / 接口说明为准。

## Engine map

| 业务名称 | engineCode | 流程组 |
|---|---|---|
| 精简编译质检 | `integratedCheck` | `product_compile` |
| 发布编译质检 | `integratedReleaseCheck` | `product_compile` |
| pb编译质检 | `pb_check` | `product_compile` |
| pb汇聚质检 | `pb_merge_check` | `product_compile` |
| pb编译 | `new_pb_compile` | `product_compile` |
| pb汇聚 | `pb_merge` | `product_compile` |
| 接边自动化 | `edge_auto` | `product_compile` |
| 接边轨迹入库 | `edge_track_post` | `product_compile` |
| 接边osm更新 | `edge_osm_update` | `product_compile` |
| 属性自动维护 | `auto_attribute_maintenance` | `product_compile` |
| BEV更新Step1 | `bev_update_large` | `byd_update` |
| BEV更新Step2 | `bev_update_large_docker_step2` | `byd_update` |
| 合规个信脱敏 | `cs_survey_compliance` | `pr` |
| BYD自动化资料导入 | `byd_automap_resource_import` | `BYD_BAMC` |
| BYD矢量数据导入 | `byd_to_osm_convertor` | `BYD_BAMC` |
| 资料准备，轨迹筛选 | `task_frame_coverage_validator` | `laser2` |
| 资料准备，step1框1 | `material-s1-frame1` | `laser2` |
| 资料准备，step1轨迹1 | `material-s1-track1` | `laser2` |
| 重跑该环节准备资料 / (内容同上，别名) | `material-s1-track1-re` | `laser2` |
| 资料准备，step1匹配1 | `material-s1-pair1` | `laser2` |
| 资料准备，step1深度学习配准 | `material-s1-dlreg` | `laser2` |
| 资料准备，step2框1 | `material-s2-frame1` | `laser2` |
| 资料准备，step2轨迹1 | `material-s2-track1` | `laser2` |
| 重跑该环节准备资料 / (内容同上，别名) | `material-s2-track1-re` | `laser2` |
| 资料准备，step2框2 | `material-s2-frame2` | `laser2` |
| 入轨成果库 | `material-align-post` | `laser2` |
| 数据前处理-JD / (engeinCode 中的dataSource 需要小写) | `import-pre-proc1-jd` | `datum_import` |
| 数据前处理-JD / (engeinCode 中的dataSource 需要小写) | `import-pre-proc1-jdm` | `datum_import` |
| 数据前处理-CAIC / (engeinCode 中的dataSource 需要小写) | `import-pre-proc1-caic` | `datum_import` |
| 数据前处理-DF / (engeinCode 中的dataSource 需要小写) | `import-pre-proc1-df` | `datum_import` |
| 数据前处理-GAC_BEV / (engeinCode 中的dataSource 需要小写) | `import-pre-proc1-gac_bev` | `datum_import` |
| 数据前处理-JD_BLA / (engeinCode 中的dataSource 需要小写) | `import-pre-proc1-jd_bla` | `datum_import` |
| 数据前处理-KDZX / (engeinCode 中的dataSource 需要小写) | `import-pre-proc1-kdzx` | `datum_import` |
| 数据前处理-HELLO_BEV / (engeinCode 中的dataSource 需要小写) | `import-pre-proc1-hello_bev` | `datum_import` |
| PLD 京东新格式 | `import-pre-proc1-pld` | `datum_import` |
| 标准化后的资料格式 | `import-pre-proc1-std` | `datum_import` |
| 数据前处理-KD | `import-nie` | `datum_import` |
|  | `import-loc_check_ppk` | `datum_import` |
|  | `import-kie` | `datum_import` |
|  | `import-rosbag_concatenator` | `datum_import` |
| 标准化 | `import-std` | `datum_import` |
| slam建图 | `import-slam` | `datum_import` |
| 轨迹入raw库 | `import-raw-post` | `datum_import` |
| 数据后处理图像 | `import-suf-proc1` | `datum_import` |
| 重跑数据后处理，补充识别资料 | `import-redo-suf-proc1` | `datum_import` |
| 数据后处理点云 | `import-suf-proc2` | `datum_import` |
| 入库后处理 | `import-raw-proc1` | `datum_import` |
| 入库后处理_标注分支选择 | `import-raw-proc-bev` | `datum_import` |
| 临近轨迹选取 | `oem-bev-nb-trk-picker` | `oem_bev` |
| BEV资料切clip | `oem-bev-tsk-frm-clipper` | `oem_bev` |
| 单框矢量准备 | `oem-bev-s1-frame1` | `oem_bev` |
| 单框矢量准备 | `oem-bev-s1-frame2` | `oem_bev` |
| 轨迹子图和匹配 | `oem-bev-s1-track1` | `oem_bev` |
| 匹配 | `oem-bev-s1-pair1` | `oem_bev` |
| 重定位 | `oem-bev-s2-posopt` | `oem_bev` |
| 复用定位结果 | `oem-bev-s2-clone-data` | `oem_bev` |
| 入成果库 | `oem-bev-align-post` | `oem_bev` |
| 轨迹视频资料 | `oem-bev-s2-track-cam` | `oem_bev` |
| 轨迹点云资料 | `oem-bev-s2-track-lidar` | `oem_bev` |
| 轨迹其它处理 | `oem-bev-s2-track-other` | `oem_bev` |
| bev作业任务资料生成 | `oem-bev-s2-frame1` | `oem_bev` |
| bev正射图识别 | `oem-bev-s2-frame2` | `oem_bev` |
| bev精修矢量数据 | `oem-bev-s2-frame3` | `oem_bev` |
| 标记聚合 | `oem-bev-s2-proc-tag` | `oem_bev` |
| 大融合（重构后) | `full_fusion` | `oem_prelabel` |
| 预标注切clip | `prelabel-s1-clip-disp` | `oem_prelabel` |
| 预标注点云处理 | `prelabel-s1-disp-lid` | `oem_prelabel` |
| 预标注SAM | `prelabel-s1-disp-sam` | `oem_prelabel` |
| 预标注矢量化 | `prelabel-s1-survey-vect` | `oem_prelabel` |
| 预标注视频处理 | `prelabel-s1-disp-cam` | `oem_prelabel` |
| 预标注PCD处理 | `prelabel-s1-disp-pcd` | `oem_prelabel` |
| 预标注正射图 | `prelabel-s1-survey-orth` | `oem_prelabel` |
| kdap规格4dbev的pmtile | `prelabel-s1-kdap-mtr` | `oem_prelabel` |
| 预标注编译KD | `prelabel-s2-kd-clip-cmpl` | `oem_prelabel` |
| 预标注编译X | `prelabel-s2-x-clip-cmpl` | `oem_prelabel` |
| 生成引擎框geojson | `prelabel-bla-mkpoly` | `oem_prelabel` |
| bla点云资料生成 | `prelabel-bla-frame1-lid` | `oem_prelabel` |
| bla点云建索引 | `prelabel-bla-track1-lid(使用COPC)` | `oem_prelabel` |
| 预标注4DBEV地图资料导入 | `prelabel-4dbev-map-imp` | `oem_prelabel` |
| 预标注4DBEV_BLA资料导入 | `prelabel-4dbev-bla-imp` | `oem_prelabel` |
| 预标注4DBEV_BEV导入 | `prelabel-4dbev-bev-imp` | `oem_prelabel` |
| 预标注4DBEV_OEM导入 | `prelabel-4dbev-oem-imp` | `oem_prelabel` |
| 预标注4DBEV_KDAP导入 | `prelabel-4dbev-kdap-imp` | `oem_prelabel` |
| BEV反投标注全库检查 | `thorough_insp_bev` | `oem_bev` |
| 编译4DBEV成果 | `compile-4dbev` | `BevCompile` |
| 点云编译后处理-jd | `cmpl-lidar-suf-proc-jd` | `pointCloudCompile` |

## Implementation recommendation

Python 侧以 `engine_map.py` 作为唯一映射源。Skill 负责告诉 Agent 何时解析流程语义；MCP Tool 只需要说明某参数或返回字段使用统一 `engineCode` 体系，无需在每个 Tool docstring 中重复完整映射。
