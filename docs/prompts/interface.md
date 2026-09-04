1. http://cs-kts.gzproduction.com/cs-kts/invoke/query
    - 请求方式：post
    - 请求参数样例：
    ```
    {
        "engineCode": "oem-bev-s2-frame1",
        "taskId": 441152
    }
    ```
    - 返回样例：/home/xumaozhou/Downloads/response_example/invoke_query_resp.json
    - 接口说明:
        - 用途：用于查询指定任务指定环节的JobInfo json
        - 返回参数说明：
            - **params**:记录流程控制参数，根据engineCode的不同里面的字段也会不一样
            - **taskSeq**:全英文的任务id
            - **tasks**:记录任务信息，为一个list，每一个list都是一个taskInfo object，其中params为任务参数，采用如下kv格式，不同类型的任务不一样
            ```
            {
                "k": "trackIds",
                "v": "kk1qHc_1782269748200"
            }
            ```

2. http://cs-kts.gzproduction.com/cs-kts/task/showByTaskSeq?taskSeq={taskSeq}
    - 请求方式：get
    - 请求参数：taskSeq
    - 返回说明：对于同一个任务，返回信息同`http://cs-kts.gzproduction.com/cs-kts/task/show?id={taaskId}`是一样的
    - 补充说明：taskId与taskSeq都是任务的唯一标识，区别是taskId是全数字的，taskSeq是全字母的


3. http://cs-kts.gzproduction.com/cs-kts/task/insertOrUpdateTags
    - 请求方式：post
    - 请求参数样例：
    ```
    {
        "taskBusinessTags": [
            {
            "id": 0,
            "k": "testKey",
            "taskId": 441152,
            "v": "testVal"
            }
        ],
        "taskIds": "441152"
    }
    ```
    - 接口说明：往指定任务信息中添加参数，添加后的字段会以kv形式出现在任务信息查询接口返回的params字段中
        - 返回值说明：code为0表示操作成功，否则表示操作失败，失败的时候message写明了失败原因，比如：
        ```
        {
            "code": 1,
            "message": "For input string: \"string\"",
            "data": null,
            "success": false
        }
        ```


4. http://cs-kts.gzproduction.com/cs-kts/labelPlatform/findRelationByCsKtsTaskId?csKtsTaskId={taskId}
    - 请求方式：get
    - 请求参数：4dBEV导入流程任务taskId
    - 返回样例：
    ```
    {
        "code": 0,
        "message": "success",
        "data": {
            "lpSrc": "pod",
            "csKtsTaskId": 440448,
            "osmTaskId": 216834957,
            "labelTaskId": 49643,
            "branchBindTaskIds": {}
        },
        "success": true
    }
    ```
    - 返回主要参数说明：
        - data.lpSrc：标注平台任务类型 labelplatformSource
        - data.csKtsTaskId：导入流程任务id
        - data.osmTaskId：矢量数据对应任务id
        - data.labelTaskId：标注平台任务id



5. http://cs-kts.gzproduction.com/cs-kts/labelPlatform/findRelationByLabelTaskId?labelTaskId={labelTaskId}&lpSrc={lpSrc}
    - 请求方式：get
    - 请求参数：标注平台任务id + 标注平台任务类型（默认取pod，不需要传参数）
    - 返回样例：
    ```
    {
        "code": 0,
        "message": "success",
        "data": {
            "lpSrc": "pod",
            "csKtsTaskId": 440448,
            "osmTaskId": 216834957,
            "labelTaskId": 49643,
            "branchBindTaskIds": {}
        },
        "success": true
    }
    ```
    - 返回主要参数说明：
        - data.lpSrc：标注平台任务类型 labelplatformSource
        - data.csKtsTaskId：导入流程任务id
        - data.osmTaskId：矢量数据对应任务id
        - data.labelTaskId：标注平台任务id



6. http://kd-material-manager.gzproduction.com/kd-material-manager/track/queryTrack?trackId={trackId}
    - 请求方式：get
    - 请求参数：轨迹id（通常 trackId.split('_')[0] 为对应入库任务的taskSeq）
    - 接口说明：用于查询已经入库轨迹的信息，如果查询为空，说明轨迹没有成功入库过
    - 返回样例：参考`/home/xumaozhou/Downloads/response_example/kk1imK_1785211909114.json`
    - 返回字段说明：需要的数据都在content.result.tracks中，其value为一个list，其中每一个obj对应该轨迹在某个pose库下的轨迹信息。以其中的字段为基准进行说明。
        - verTimestamps：记录当前pose库中轨迹的版本号，毫秒时间戳
        - taskId：对应入库任务的taskSeq
        - poseTag：对应pose库名称
        - geometry:WTK格式的轨迹形态
        - totalMileage：轨迹总长度
        - otherFiles：hdfs杂项数据存放目录，其往上一级即为对应入库数据hdfs存储根目录
        - previousTrackId：同一批数据下前一个轨迹id（允许为空）
        - nextTrackId：同一批数据下后一个轨迹id（允许为空）
        - sourceTag：轨迹来源标识


7. http://cs-argo-dispatcher.gzproduction.com/cs-argo-dispatcher/job/findDomainsByTaskSeq?taskSeq={taskSeq}
    - 请求方式：get
    - 请求参数：taskSeq
    - 接口说明：根据任务id查询该任务各个环节的运行信息
    - 返回样例：/home/xumaozhou/Downloads/response_example/findDomainsByTaskSeq_resp.json
    - 返回字段说明：content.data为一个list，每一个obj为流程中指定环节的JobInfo，每个JobInfo中需要关注的字段如下
        - taskBusinessId：含义为对应Job的taskId
        - taskSeq：对应Job的taskSeq
        - engineCode：对应环节的 engineCode
        - state：任务状态，"0"表示任务在排队，"1"表示任务运行中，"4"表示任务运行失败，"9"表示任务运行成功，其他取值表示任务异常
        - tasks：为一个list，一个Job可能包含多个Task，这个list中记录的为对应任务的taskInfo
            - state：任务状态，"0"表示任务在排队，"1"表示任务运行中，"4"表示任务运行失败，"9"表示任务运行成功，其他取值表示任务异常
            - podName：k8s中的pod name，可用于查询pod日志
            - workflowName：k8s中的workflowName，可用于查询日志
            - stateAt： 任务开始时间，毫秒时间戳
            - params：任务参数，不同任务可能不一样（其中字段与csKtsShowByTaskId接口返回的定义一致）


8. http://cs-argo-dispatcher.gzproduction.com/cs-argo-dispatcher/job/findDomainsByTaskId?taskId={taskId}
对于相同任务，查询结果与前一个接口保持一致，区别是使用taskSeq查询还是taskId查询
