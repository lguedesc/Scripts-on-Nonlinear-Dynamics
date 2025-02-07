class PoincareMap:
    def __init__(self, nP, nDiv, t0, dt, x0, func, p, nP_transient_end):
        self.nP = nP
        self.nDiv = nDiv
        self.t0 = t0
        self.dt = dt
        self.x0 = np.array(x0)
        self.func = func
        self.p = p
        self.nP_transient_end = nP_transient_end
        
        # Allocate result arrays
        self.int_result = np.zeros((nP * nDiv + 1, len(x0) + 1))
        self.poinc_result = np.zeros((nP - nP_transient_end, len(x0) + 1))
        
        # Initialize first row
        self.int_result[0, 0] = t0
        self.int_result[0, 1:] = x0
    
    def poincare_map_periodic_excitation(self, current_P, current_Div, current_step, pp, section=1):
        if current_P >= self.nP_transient_end and current_Div == section:
            self.poinc_result[pp, :] = self.int_result[current_step, :]
            pp += 1
        return pp
    
    def integrate_and_compute_poincare(self):
        i, pp = 0, 0
        for j in range(self.nP):
            for k in range(self.nDiv):
                self.int_result[i + 1, 1:] = rk4(self.func, self.int_result[i, 1:], self.dt, 
                                                 self.int_result[i, 0], self.p)
                pp = self.poincare_map_periodic_excitation(j, k, i, pp)
                self.int_result[i + 1, 0] = self.int_result[i, 0] + self.dt
                i += 1
        return self.int_result, self.poinc_result