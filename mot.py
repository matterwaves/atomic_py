from scipy.special import ellipk as K
from scipy.special import ellipe as E
from numpy import pi,e,sqrt,sin,cos,exp,log
from matplotlib import pyplot as plt
import numpy as np
import copy


class laser():
    """
    Ignore the intensity profile for now
    Only consider intensity, polarization, and direction
    """
    def __init__(self,intensity=0,polarization=+1,direction=np.array([1,0,0])):
        """
        intensity: float [mW/cm^2]
        polarization: +1 or -1  [RHC or LHC] helicity
        direction: np.array([float]), unit vector
        """
        assert intensity >0, "Intensity must be a positive number"
        assert polarization==1 or polarization == -1, "polarization must be +-1"
        self.intensity=intensity
        self.polarization=polarization
        self.direction=direction/np.linalg.norm(direction)

    def __repr__(self):
        string=f"Intensity={self.intensity:.2f} mW/cm^2\n"
        string+=f"helicity={self.polarization}\n"
        string+=f"direction={np.array2string(self.direction,precision=2)}\n"
        return string
    def __str__(self):
        return self.__repr__()


    def reflect(self,reflectivity=1,normal_direction=[1,0,0]):
        """
        Reflect a laser off an ordinary mirror,
        requires mirror angle (relative to laser direction)
        returns a new laser object representing the reflected beam
        """
        assert reflectivity>0 and reflectivity<=1, "Reflectivity must be between 0 and 1"
        assert np.shape(normal_direction)==np.shape(np.array([0,0,0])), "direction must be a 3-vector"
        normal_direction=normal_direction/np.linalg.norm(normal_direction)

        # Invert the vector component which is normal to the mirror surface
        normal_component=np.dot(self.direction,normal_direction)*normal_direction
        reflected_direction=self.direction-2*normal_component

        return laser(self.intensity*reflectivity,polarization=self.polarization*-1,direction=reflected_direction)

    def diffract(self,efficiency=[0,1],bragg_direction=[1,0,0]):
        """
        Same as reflection, except there is an angle-dependent intensity increase
        returns a new laser object representing the diffracted beam
        Assume that the beam is at normal incidence to the diffraction grating
        """
        assert np.shape(bragg_direction)==np.shape(np.array([0,0,0])), "direction must be a 3-vector"
        bragg_direction=bragg_direction/np.linalg.norm(bragg_direction)
        ## zero order diffraction looks like a mirror at normal incidence, with low reflectivity
        order0=self.reflect(efficiency[0],normal_direction=self.direction)

        ## Bragg angle is twice the effective angle of incidence
        bragg_angle=2*np.arcsin(sqrt(1-np.dot( self.direction,bragg_direction)**2 ))

        ## First order diffraction has the angle-dependent intensity boost
        order1=self.reflect(efficiency[1]/sin(bragg_angle),normal_direction=bragg_direction)

        return order0,order1

    def gain(self,gain):
        self.intensity*=gain


class magneticCoil():
    """
    A magnetic coil, used for generating fields in a MOT
    """
    mu0=1.25663706212e-6#N/A^2

    def __init__(self,diameter=1,turns=1,current=1,origin=np.array([0,0,0])):
        """
        diameter [meters]
        turns [number]
        current [amps]
        origin=[meters,meters] center position of coil
        """
        assert diameter >0 , "Diameter must be positive"
        assert type(turns) == int, "turns must be an integer"
        origin=np.array(origin)
        assert np.shape(origin)==np.shape(np.array([0,0,0]))
        self.diameter=diameter
        self.turns=turns
        self.current=current
        self.origin=origin.astype(np.float64)

    def __repr__(self):
        string =f"Diameter: {self.diameter:0.2f} meters"
        string+=f"\n turns*current: {self.turns}*{self.current:0.2f} amps"
        string+=f"\n origin: {self.origin}"
        return string

    def __str__(self):
        return self.__repr__()
    def shift(self,displacement=[0,0,0]):
        displacement=np.array(displacement)
        assert np.shape(displacement) == np.shape(np.array([0,0,0]))
        self.origin+=np.array(displacement)

    def field(self,*coords):
        """
        magnetic field due to the coil, when current I passes thru it
        r = [meters]
        z = [meters]
        field = [Gauss]
        fully vectorized!
        """
        ## Make sure all inputs are arrays to avoid index issues
        x,y,z=coords[0],coords[1],coords[2]
        if type(x) != type(np.array([])):
            x=np.array(x).astype(np.float64)
        if type(y) != type(np.array([])):
            y=np.array(y).astype(np.float64)
        if type(z) != type(np.array([])):
            z=np.array(z).astype(np.float64)

        ## Transform from cartesian to cylindrical coordinates
        ## Shift r,z coordinates relative to loop origing
        r=sqrt((x-self.origin[0])**2+(y-self.origin[1])**2)
        ## Convert from meters to units of loop radius
        r0=2*r/self.diameter
        z0=2*(z-self.origin[2])/self.diameter

        ## Define useful constants
        I=self.turns*self.current
        B0=1e4*I*magneticCoil.mu0/self.diameter ##[gauss] Field strength at center of loop
        Q=(1+r0)**2 + z0**2
        m=4*r0/Q

        ## Analytic formula for magnetif field
        Bz=B0/(pi*sqrt(Q))*( E(m)*(1-r0**2-z0**2)/(Q-4*r0)+K(m))

        ## for scalar r, avoid indexing
        if np.shape(r0) == ():
            if r0==0:
                Br=0
            else:
                Br=B0/(pi*sqrt(Q))*(z0/r0)*( E(m)*(1+r0**2+z0**2)/(Q-4*r0)-K(m))
        else:
            ## Keep track of where r0==0 to avoid div by 0
            r_reg=np.copy(r0)
            reg_idx=np.where(r0==0)
            r_reg[reg_idx]=1
            Br=B0/(pi*sqrt(Q))*(z0/r_reg)*( E(m)*(1+r0**2+z0**2)/(Q-4*r0)-K(m))
            ## force B_r=0 on axis
            Br[reg_idx]=0

        ##Transform back to cartesian coordinates
        ## x/r = cos(theta), y/r = sin(theta)
        Bx=Br*x/r
        By=Br*y/r
        return Bx,By,Bz # Gauss
        #return Br,Bz ## Gauss

    def fieldOnAxis(self,z):
        """
        Analytic form for on-axis field strength
        returns strength in gauss
        """
        I=self.turns*self.current
        ## Shift z coordinates relative to loop origing
        z -= self.origin[1]
        ## Convert to units of loop radius
        r=self.diameter/2
        return 1e4*magneticCoil.mu0/(4*pi) *(2*pi*r**2*I)/(z**2+r**2)**(3/2)

    def consistencyCheck(self):
        z=np.linspace(-10,10,50)
        _,Bz=self.field(np.zeros(np.shape(z)),z)
        Bz_=self.fieldOnAxis(z)
        err=2*(Bz-Bz_)/(Bz+Bz_)
        print("Testing analytic off axis formula against on-axis formula")
        print(f"Maximum relative error: {np.max(err)}")
        print(f"Mean relative error: {np.mean(err)}")
        if np.max(err) < 10**10:
            print("TEST PASSED")

    def copy(self):
        return copy.deepcopy(self)

class coilSystem():
    def __init__(self,coils=None):
        if coils is None:
            self.coils=[]
        elif type(coils)==type(magneticCoil()):
            self.coils=[coils]
        else:
            self.coils=coils
        for coil in self.coils:
            assert type(coil)==type(magneticCoil()), "All coils must be of type magneticCoil"
        self.r_range,self.z_range=(-1,1,40),(-1,1,40)
        #self.rs,self.zs=np.meshgrid(np.linspace(*self.r_range),np.linspace(*self.z_range))
        self.xs,self.ys,self.zs=np.meshgrid(np.linspace(*self.r_range),np.linspace(*self.r_range),np.linspace(*self.z_range))
        return

    def __add__(self,other):
        return coilSystem(self.coils+other.coils)
    def __repr__(self):
        string=""
        for num,coil in enumerate(self.coils):
            string+=f"Coil #{num}:\n"
            string+=repr(coil)
            string+="\n"
        return string
    def __str__(self):
        return self.__repr__()
    def field(self,*coords):
        return sum([np.array(coil.field(*coords)) for coil in self.coils])

    def setROI(self,r_range=(-1,1,40j),z_range=(-1,1,40j)):
        self.r_range=r_range
        self.z_range=z_range
        #self.rs,self.zs=np.meshgrid(np.linspace(*self.r_range),np.linspace(*self.z_range))
        self.xs,self.ys,self.zs=np.mgrid[slice(*self.r_range),slice(*self.r_range),slice(*self.z_range)]

    def view(self,fieldFunc=None ):

        if fieldFunc is None:
            fieldFunc=self.field
        #Bx,By,Bz=self.field(self.xs,self.ys,self.zs)
        Bx,By,Bz=fieldFunc(self.xs,self.ys,self.zs)

        ## Take a cross-section for viewing
        mask=self.ys==np.min(np.abs(self.ys))
        middle_idx=int(np.shape(self.xs)[0]/2)

        Bx=Bx[middle_idx,:,:]
        By=By[middle_idx,:,:]
        Bz=Bz[middle_idx,:,:]
        zs,ys=self.zs[middle_idx,:,:],self.ys[middle_idx,:,:]

        mag=sqrt(Bx**2+By**2+Bz**2)

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ## Plotting in terms of z,y because of streamplot indexing requiremens
        ax.streamplot(zs,ys,Bz,By,density=1.5,color=mag)
        ax.set(xlim=(ys.min(),ys.max() ), ylim=(ys.min(),zs.max()),xlabel="Z (m)", ylabel="Y (m)" )
        plt.show()
        return


    def gradient(self):
        """
        Br,Bz in Gauss
        r,z in meters
        Return [[Bxx, Bxy, Bxz],
                [Byx, Byy, Byz],
                [Bzx, Bzy, Bzz]]
            in Gauss/meter
            Evaluated at the field minimum

        Matrix should be traceless and antisymmetric
        """

        Bx,By,Bz=self.field(self.xs,self.ys,self.zs)
        mag=sqrt(Bx**2+By**2+Bz**2)
        min_idx=np.unravel_index(np.argmin(mag),np.shape(mag))
        print(min_idx)

        print(f"Field minimum @ x={self.xs[min_idx]:.3f},y={self.ys[min_idx]:.3f}, z={self.zs[min_idx]:.3f}")
        print(f"B={mag[min_idx]:.2f} Gauss ")

        dx,dy,dz=self.xs[1,0,0]-self.xs[0,0,0],self.ys[0,1,0]-self.ys[0,0,0], self.zs[0,0,1]-self.zs[0,0,0]
        Grad_Bx=np.array(np.gradient(Bx,dy,dx,dz))
        Grad_By=np.array(np.gradient(By,dy,dx,dz))
        Grad_Bz=np.array(np.gradient(Bz,dy,dx,dz))



        Grad_B= np.vstack((
            Grad_Bx[(slice(None),*min_idx)],
            Grad_By[(slice(None),*min_idx)],
            Grad_Bz[(slice(None),*min_idx)])
            )
        return Grad_B/100 ## Gauss/cm

    def symmetry_check(self, Grad_B):
        """
        Grad_B should be traceless and antisymmetric
        """

        ## compute the mean abs eigenvalues
        ## to provide a size scale to compare against
        eigs,_= np.linalg.eig(Grad_B)
        size=sum(np.abs(eigs))/3

        trace_err=np.trace(Grad_B)/size
        symm_eigs,_=np.linalg.eig( Grad_B-Grad_B.transpose()  )
        symm_err=sum(np.abs(symm_eigs))/(3*size)

        print(f"Relative trace error: {trace_err:.2E}")
        print(f"Relative symmetry error: {symm_err:.2E}")
        return {"symm_error": symm_err, "trace_error":trace_err}

class mot():
    """
    A mot traps atoms using the scattering force due to doppler shifts and magnetic fields
    The doppler term is determined only by the laser power and geometry
    The magnetic term is determined by the laser power, geometry, polarization, and applied field profile
    """

    def __init__(self,coils,lasers,transition,detuning=10):
        self.coils=coils
        self.lasers=lasers
        self.transition=transition
        self.detuning=detuning


    def __repr__(self):
        string="*****MOT Class instance*****\n"
        string+="***** Coil System *****\n"
        string+=str(self.coils)
        string+="***** Laser system *****\n"
        for laser in self.lasers:
            string+=str(laser)
        string+=f'detuning = {self.detuning:.2f}\n'
        string+="***** Atomic transition *****\n"
        string+=str(self.transition)
        return string

    def __str__(self):
        return self.__repr__()

    @property
    def balance(self):
        return sum( [laser.intensity*laser.direction for laser in self.lasers] )

    @property
    def v_mat(self):
        """
        doppler structure matrix
        mat = sum:  v *outer* v /norm(v)
        """
        return sum([np.outer(v,v)/np.linalg.norm(v) for v in [laser.direction for laser in self.lasers]  ])

    @property
    def b_mat(self):
        """
        magnetic structure matrix
        mat = sum:  polz*v *outer* v /norm(v)
        """
        return sum([s*np.outer(v,v)/np.linalg.norm(v) for (v,s) in [(laser.direction,laser.polarization) for laser in self.lasers]   ])

### Dimensionless MOT parameters
    @property
    def betaTot(self):
        """
        Sum of all intensities, in units of saturation intensity
        Used for computed saturation effects
        """
        return sum([laser.intensity for laser in self.lasers])/self.transition.Isat

    @property
    def KK(self):
        """
        Lorentzian factor for bare detuning
        """
        return 1/(1+self.betaTot+4*(self.detuning/self.transition.linewidth)**2)

    @property
    def C(self):
        """
        useful constant.  What is the interpretation?
        """
        return 8*(self.detuning/self.transition.linewidth)*self.KK**2


    @property
    def B0(self):
        """
        Effective offset magnetic field caused by radiation imbalance
        """

        return -1*self.KK/(self.C*self.transition.MOT_magnetic)* np.matmul(np.linalg.inv(self.b_mat),self.balance)

    def acceleration(self,*coords):
        """
        Calculate the MOT acceleration at position coords
        Fully vectorized
        """
        effective_magnetic=self.coils.field(*coords)+np.reshape(self.B0,(3,1,1,1))
        return -1*self.transition.a*self.transition.MOT_magnetic*self.C*np.einsum("ij,jklm",self.b_mat,effective_magnetic)

    def view(self):
        self.coils.view(fieldFunc=self.acceleration)
