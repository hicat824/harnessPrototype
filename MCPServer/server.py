from mcp.server import MCPServer

from argoDispatcherInterface import (
    csArgoFindDomainsByTaskId,
    csArgoFindDomainsByTaskSeq,
)
from csKtsInterface import (
    csKtsFindRelationByCsKtsTaskId,
    csKtsFindRelationByLabelTaskId,
    csKtsInsertOrUpdateTags,
    csKtsInvokeQuery,
    csKtsShowByTaskId,
    csKtsShowByTaskSeq,
)
from kdMaterialInterface import kdMaterialQueryTrack
from resultCache import readCachedResult

mcp = MCPServer("KD-tools")
mcp.add_tool(csKtsShowByTaskId)
mcp.add_tool(csKtsShowByTaskSeq)
mcp.add_tool(csKtsInvokeQuery)
mcp.add_tool(csKtsInsertOrUpdateTags)
mcp.add_tool(csKtsFindRelationByCsKtsTaskId)
mcp.add_tool(csKtsFindRelationByLabelTaskId)
mcp.add_tool(kdMaterialQueryTrack)
mcp.add_tool(csArgoFindDomainsByTaskSeq)
mcp.add_tool(csArgoFindDomainsByTaskId)
mcp.add_tool(readCachedResult)

if __name__ == "__main__":
    mcp.run()
