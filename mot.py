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


    def reflect(self):
        """
        Reflect a laser off an ordinary mirror,
        requires mirror angle (relative to laser direction)
        returns a new laser object representing the reflected beam
        """

    def diffract(self):
        """
        Same as reflection, except there is an angle-dependent intensity increase
        returns a new laser object representing the diffracted beam
        """

    def gain(self,gain):
        self.intensity*=gain


class magneticCoil():
    """
    A magnetic coil, used for generating fields in a MOT
    """
    mu0=1.25663706212e-6#N/A^2

    def __init__(self,diameter=1,turns=1,current=1,origin=np.array([0,0])):
        """
        diameter [meters]
        turns [number]
        current [amps]
        origin=[meters,meters] center position of coil
        """
        assert diameter >0 , "Diameter must be positive"
        assert type(turns) == int, "turns must be an integer"
        origin=np.array(origin)
        assert np.shape(origin)==np.shape(np.array([0,0]))
        self.diameter=diameter
        self.turns=turns
        self.current=current
        self.origin=origin.astype(np.float64)

    def __repr__(self):
        string =f"Diameter: {self.diameter:0.2f} meters"
        string+=f"\n turns*current: {self.turns}*{self.current:0.2f} amps"
        string+=f"\n origin: {self.origin}"
        return string

    def shift(self,displacement=[0,0]):
        displacement=np.array(displacement)
        assert np.shape(displacement) == np.shape(np.array([0,0]))
        self.origin+=np.array(displacement)

    def field(self,r,z):
        """
        magnetic field due to the coil, when current I passes thru it
        r = [meters]
        z = [meters]
        field = [Gauss]
        fully vectorized!
        """
        ## Make sure all inputs are arrays to avoid index issues
        if type(r) != type(np.array([])):
            r=np.array(r).astype(np.float64)
        if type(z) != type(np.array([])):
            z=np.array(z).astype(np.float64)

        ## Shift r,z coordinates relative to loop origing
        r0 = r-self.origin[0]
        z0 = z-self.origin[1]
        ## Convert from meters to units of loop radius
        r0=2*r0/self.diameter
        z0=2*z0/self.diameter

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
        return Br,Bz ## Gauss

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
        self.rs,self.zs=np.meshgrid(np.linspace(*self.r_range),np.linspace(*self.z_range))
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
    def field(self,r,z):
        return sum([np.array(coil.field(r,z)) for coil in self.coils])

    def setROI(self,r_range=(-1,1,40),z_range=(-1,1,40)):
        self.r_range=r_range
        self.z_range=z_range
        self.rs,self.zs=np.meshgrid(np.linspace(*self.r_range),np.linspace(*self.z_range))

    def view(self):
        Br,Bz=self.field(self.rs,self.zs)
        mag=sqrt(Br**2+Bz**2)
        fig=plt.figure()
        ax=fig.add_subplot(111)

        ax.streamplot(self.rs,self.zs,Br,Bz,density=1.5,color=mag)
        ax.set(xlim=(self.rs.min(),self.rs.max() ), ylim=(self.zs.min(),self.zs.max()) )
        plt.show()


    def gradient(self):
        """
        Br,Bz in Gauss
        r,z in meters
        Return [[Brr, Brz],
                [Bzr, Bzz]]
            in Gauss/meter
            Evaluated at the field minimum
        """

        Br,Bz=self.field(self.rs,self.zs)
        mag=sqrt(Br**2+Bz**2)
        min_idx=np.unravel_index(np.argmin(mag),np.shape(mag))

        print(f"Field minimum @ r={self.rs[min_idx]}, z={self.zs[min_idx]}")
        dr,dz=self.rs[0,1]-self.rs[0,0],self.zs[1,0]-self.zs[0,0]
        Grad_Br=np.gradient(Br,dz,dr)
        Grad_Bz=np.gradient(Bz,dz,dr)
        Grad_B=np.array([[Grad_Br[1][min_idx],Grad_Br[0][min_idx]],
                    [Grad_Bz[1][min_idx],Grad_Bz[0][min_idx]]]) ## Gauss/meter

        return Grad_B/100 # Gauss/cm




class mot():
    """
    A mot traps atoms using the scattering force due to doppler shifts and magnetic fields
    The doppler term is determined only by the laser power and geometry
    The magnetic term is determined by the laser power, geometry, polarization, and applied field profile
    """

    def __init__(self,lasers):
        pass

