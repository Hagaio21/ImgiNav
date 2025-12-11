"""
DataFlow - Dict-like object for tracking data flow through model components.
"""
import torch
from typing import Any, Dict, Optional, List, Tuple


class DataFlow(dict):
    """
    Dict-like object that tracks data flow through components.
    Fully backward compatible with dict operations.
    """
    
    def __init__(
        self, 
        data: Optional[Dict[str, Any]] = None, 
        source_component: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(data or {})
        self.source_component = source_component
        self.metadata = metadata or {}
        self._access_log: List[Tuple[str, str]] = []
        self._shape_info: Dict[str, Tuple] = {}
        self._consumed_by: Optional[str] = None
        self._input_from: Optional[str] = None
        
        if data:
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    self._shape_info[key] = tuple(value.shape)
    
    def __getitem__(self, key: str) -> Any:
        self._access_log.append(('getitem', key))
        return super().__getitem__(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        self._access_log.append(('get', key))
        return super().get(key, default)
    
    def __contains__(self, key: str) -> bool:
        self._access_log.append(('contains', key))
        return super().__contains__(key)
    
    def keys(self):
        self._access_log.append(('keys', None))
        return super().keys()
    
    def values(self):
        self._access_log.append(('values', None))
        return super().values()
    
    def items(self):
        self._access_log.append(('items', None))
        return super().items()
    
    def update(self, other: Dict[str, Any] = None, **kwargs) -> None:
        if other:
            if isinstance(other, DataFlow):
                super().update(other)
                self._shape_info.update(other._shape_info)
                if other.source_component:
                    self._input_from = other.source_component
            else:
                super().update(other)
                for key, value in other.items():
                    if isinstance(value, torch.Tensor):
                        self._shape_info[key] = tuple(value.shape)
        
        if kwargs:
            super().update(kwargs)
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    self._shape_info[key] = tuple(value.shape)
    
    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if isinstance(value, torch.Tensor):
            self._shape_info[key] = tuple(value.shape)
    
    def mark_consumed_by(self, component_name: str) -> None:
        self._consumed_by = component_name
    
    def get_shape(self, key: str) -> Optional[Tuple]:
        return self._shape_info.get(key)
    
    def get_all_shapes(self) -> Dict[str, Tuple]:
        return self._shape_info.copy()
    
    def get_access_log(self) -> List[Tuple[str, str]]:
        return self._access_log.copy()
    
    def get_flow_info(self) -> Dict[str, Any]:
        return {
            "source": self.source_component,
            "consumed_by": self._consumed_by,
            "input_from": self._input_from,
            "keys": list(self.keys()),
            "shapes": self.get_all_shapes()
        }
    
    def copy(self) -> 'DataFlow':
        """Create a copy of this DataFlow."""
        new_df = DataFlow(
            data=dict(self),
            source_component=self.source_component,
            metadata=self.metadata.copy()
        )
        new_df._shape_info = self._shape_info.copy()
        new_df._access_log = self._access_log.copy()
        new_df._consumed_by = self._consumed_by
        new_df._input_from = self._input_from
        return new_df
    
    def to_dict(self) -> Dict[str, Any]:
        return dict(self)
    
    @classmethod
    def wrap(cls, data: Any, key: str = "data", source_component: Optional[str] = None) -> 'DataFlow':
        return cls(data={key: data}, source_component=source_component)

