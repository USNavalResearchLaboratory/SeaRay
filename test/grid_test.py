import pytest
import numpy as np
import modules.grid_tools as grid_tools

class TestGrids:

    def test_cyclic(self):
        nodes = grid_tools.cyclic_nodes(0.0,5.0,5)
        assert np.allclose(nodes,[0.0,1.0,2.0,3.0,4.0])
        center,width = grid_tools.cyclic_center_and_width(0.0,5.0)
        assert center == pytest.approx(2.5)
        assert width == pytest.approx(5)
    
    def test_normal(self):
        nodes = grid_tools.cell_centers(0.0,5.0,5)
        assert np.allclose(nodes,[0.5,1.5,2.5,3.5,4.5])
        walls = grid_tools.cell_walls(0.5,4.5,5)
        assert np.allclose(walls,[0.0,1.0,2.0,3.0,4.0,5.0])

    def test_hypersurface(self):
        a = np.zeros((4,4,4))
        idx = grid_tools.hypersurface_idx(a,0,1)
        assert idx == ((1,),slice(None),slice(None))

    def test_ghost_cells(self):
        a = np.zeros((4,4,4))
        nodes = grid_tools.cell_centers(0.0,4.0,4)
        xa,xnodes = grid_tools.AddGhostCells(a,nodes,1)
        assert np.allclose(xa,np.zeros((4,5,4)))
        assert np.allclose(xnodes,[-0.5,0.5,1.5,2.5,3.5])

class TestInterpolation:

    def test_full_slab_interior(self):
        a = np.ones((4,4,4))
        a[2,2,2] = 2.0
        wn = grid_tools.cyclic_nodes(0.0,4.0,4)
        xn = grid_tools.cell_centers(0.0,4.0,4)
        yn = grid_tools.cell_centers(0.0,4.0,4)
        w = np.array([2.0,2.0])
        x = np.array([2.5,2.5])
        y = np.array([2.5,3.0])
        data = grid_tools.DataFromGrid(w,x,y,wn,xn,yn,a)
        assert np.allclose(data,[2.0,1.5])

    def test_full_slab_edge(self):
        a = np.ones((4,4,4))
        a[2,2,3] = 2.0
        wn = grid_tools.cyclic_nodes(0.0,4.0,4)
        xn = grid_tools.cell_centers(0.0,4.0,4)
        yn = grid_tools.cell_centers(0.0,4.0,4)
        w = np.array([2.0,2.0,2.0])
        x = np.array([2.5,2.5,2.5])
        y = np.array([4.0,3.5,3.0])
        data = grid_tools.DataFromGrid(w,x,y,wn,xn,yn,a)
        assert np.allclose(data,[2.5,2.0,1.5])
