__all__ = ["FinancialGraphService", "graph_service"]


def __getattr__(name):
    if name in {"FinancialGraphService", "graph_service"}:
        from src.graph.service import FinancialGraphService, graph_service

        return {
            "FinancialGraphService": FinancialGraphService,
            "graph_service": graph_service,
        }[name]
    raise AttributeError(name)
