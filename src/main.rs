
// Compilar y ejecutar con "cargo run"
// Compilar con "cargo build"
// Ejecutar con "./target/debud/redneuronal"

extern crate rand;
use rand::Rng;

use std::fs::File;
use std::io::Read;
use std::mem;


struct Neuron
{
	pesos: Vec<f32>,
	error: f32,
	salida: f32
}

fn sigmoide(x: f32) -> f32
{	// 1/(1+Math.pow(Math.E, -input))
	let mut s = 1f32 / (1f32 + (-x).exp());
	return s;
}

fn derivadaSigmoide(o: f32) -> f32
{
	let mut s = o * (1f32 - o);
	return s;
}

struct RedNeuronal {
    capas: Vec<Vec<Neuron> >,
    tasaAprendizaje: f32
}


impl RedNeuronal
{
	fn new(entradas:i32, capasOcultas:i32, ocultas:i32, salidas:i32, tasa:f32) -> RedNeuronal
	{
		// creamos la red
	    let mut nn = Vec::new();

	    // creamos la capa de entrada
	    nn.push(Vec::new());
	    for i in 0..entradas
	    {
	    	let x = rand::random::<f32>();
	    	nn[0].push( Neuron{pesos:vec![x], error:0.0, salida:1.0} );
	    }

	    // creamos la capa(s) oculta(s)
	    for c in 0..capasOcultas
	    {
	    	//println!("capa oculta {}", c);
		    nn.push(Vec::new());
		    for i in 0..ocultas
		    {
		    	let mut pesos = vec![];
		    	let entradasACapa = nn[nn.len()-2].len();
		    	//println!("entradas {}", entradasACapa);
		    	for j in 0..entradasACapa
		    	{
		    		//println!("peso {}", j);
		    		let x = rand::random::<f32>();
		    		pesos.push(x);
		    	}
		    	let capaAct = nn.len()-1;
		    	nn[capaAct].push( Neuron{pesos:pesos, error:0.0, salida:1.0} );
		    }
		}

	    // creamos la capa de salida
	    nn.push(Vec::new());
	    for i in 0..salidas
	    {
	    	let mut pesos = vec![];
	    	let entradasACapa = nn[nn.len()-2].len();
	    	for j in 0..entradasACapa
	    	{
	    		let x = rand::random::<f32>();
	    		pesos.push(x);
	    	}
	    	let capaAct = nn.len()-1;
	    	nn[capaAct].push( Neuron{pesos:pesos, error:0.0, salida:1.0} );
	    }

	    let mut red = RedNeuronal{ capas: nn, tasaAprendizaje: tasa};

	    return red;
	}

	fn ejecutar(&mut self, entrada: &Vec<f32>) -> &mut RedNeuronal
	{
		let mut o: f32;
		// calculamos capa entrada
		for neuron in 0..self.capas[0].len()
		{
			o = self.capas[0][neuron].pesos[0] * entrada[neuron];
			self.capas[0][neuron].salida = sigmoide(o);
		}

		// calculamos capa oculta y salida
		for capa in 1..self.capas.len()
		{
			for neuron in 0..self.capas[capa].len()
			{
				o = 0.0;
				for p in 0..self.capas[capa][neuron].pesos.len()
				{
					o = o + ( self.capas[capa][neuron].pesos[p] * self.capas[capa-1][p].salida );
				}
				self.capas[capa][neuron].salida = sigmoide(o);
			}
		}

		return self;
	}

	fn print(&mut self) -> &mut RedNeuronal
	{
		let o = 0.0;
		// calculamos capa entrada
		for c in 0..self.capas.len()
		{
			println!("CAPA {}", c);
			for n in 0..self.capas[c].len()
			{
				println!("\tNEURON {}", n);
				for p in 0..self.capas[c][n].pesos.len()
				{
					println!("\tPeso {} // Salida {}", self.capas[c][n].pesos[p], self.capas[c][n].salida);
				}
			}
		}

		return self;
	}

	fn entrenarBackPropagation(&mut self, entradas: &Vec<Vec<f32> >, salidas: &Vec<Vec<f32> >, epocas:i32) -> &mut RedNeuronal
	{
		for epoca in 0..epocas
		{
			for entrada in 0..entradas.len()
			{
				// Ejecuto la red
				self.ejecutar(&entradas[entrada]);

				//println!("ENTRADA: {}", entrada);

				// Ahora, calculamos el error de la capa de salida y actualizamos sus pesos
				let mut capaSalida = self.capas.len()-1;
				for neuron in 0..self.capas[capaSalida].len()
				{
					self.capas[capaSalida][neuron].error = (salidas[entrada][neuron] - self.capas[capaSalida][neuron].salida) * derivadaSigmoide(self.capas[capaSalida][neuron].salida);

					for peso in 0..self.capas[capaSalida][neuron].pesos.len()
					{
						//println!("neuron {} / peso {} / pesos {} / neuronas ant {}", neuron, peso, self.capas[capaSalida][neuron].pesos.len(), self.capas[capaSalida-1].len());
						self.capas[capaSalida][neuron].pesos[peso] += self.tasaAprendizaje * self.capas[capaSalida][neuron].error * self.capas[capaSalida-1][peso].salida;
					}
				}
				//println!("Aqui");
				// Ahora, calculamos el error de la capa de oculta y actualizamos sus pesos
				//let mut capaOculta = 1;
				for capaOculta in 1..self.capas.len()-1
				{
					for neuron in 0..self.capas[capaOculta].len()
					{
						// el error de esta neurona sera: error de cada neurona de la siguiente capa * cada peso (acumulado)
						let mut errorAcumulado = 0f32;
						let capaSiguiente = capaOculta + 1;
						let capaAnterior = capaOculta - 1;
						for n in 0..self.capas[capaSiguiente].len()
						{
							for peso in 0..self.capas[capaOculta][neuron].pesos.len()
							{
								errorAcumulado += self.capas[capaSiguiente][n].error * self.capas[capaOculta][neuron].pesos[peso];
							}
						}
						// Finalmente, el error del neuron sera el acumulado * derivada de la funcion para la salida de este neuron
						self.capas[capaOculta][neuron].error = errorAcumulado * derivadaSigmoide(self.capas[capaOculta][neuron].salida);
						
						// Una vez que tenemos el error propagado a este neuron, calculamos los nuevos pesos
						for peso in 0..self.capas[capaOculta][neuron].pesos.len()
						{
							self.capas[capaOculta][neuron].pesos[peso] += self.tasaAprendizaje * self.capas[capaOculta][neuron].error * self.capas[capaAnterior][peso].salida;
						} 
					}
				}
			}
		}

		return self;
	}

	fn printSalida(&mut self) -> &mut RedNeuronal
	{
		let ultimaCapa = self.capas.len()-1;
		for neuron in 0..self.capas[ultimaCapa].len()
		{
			println!("Salida {}: {}", neuron, self.capas[ultimaCapa][neuron].salida);
		}

		return self;
	}

	fn salida(&mut self) -> Vec<f32>
	{
		let ultimaCapa = self.capas.len()-1;
		let mut salida: Vec<f32> = vec![];

		for neuron in 0..self.capas[ultimaCapa].len()
		{
			salida.push(self.capas[ultimaCapa][neuron].salida);
		}

		return salida;
	}
}


fn invertir(en: &mut [u8])
{
	for i in 0..en.len()
	{
		println!("{}", i);
	}
}

fn main()
{

	// entradas y salidas
	let ent: Vec<Vec<f32> > = vec![vec![0f32, 1f32], vec![1f32, 0f32]];
	let sal: Vec<Vec<f32> > = vec![vec![0f32],vec![1f32]];

	let entradas = ent[0].len() as i32;
	let salidas = sal[0].len() as i32;
	let neuronasOcultas = 5;
	let capasOcultas = 2;
	let epocas = 20;

	let mut nn = RedNeuronal::new(entradas, capasOcultas, neuronasOcultas, salidas, 0.6);

	// Salidas sin entrenar
	println!("Salidas antes de entrenar:");
	nn.ejecutar(&ent[0]);
	nn.printSalida();
	nn.ejecutar(&ent[1]);
	nn.printSalida();

	nn.entrenarBackPropagation(&ent, &sal, epocas);
	
	// salidas entrenadas
	println!("Salidas despues de entrenar:");
	nn.ejecutar(&ent[0]);
	nn.printSalida();
	nn.ejecutar(&ent[1]);
	nn.printSalida();

	//let mut s = nn.salida();
	//println!("{}", s[0]);


	// Lectura de ficheros
	let mut file=File::open("data/train_images").unwrap();
    let mut buf = [0; 4];
    file.read(&mut buf);
    invertir(&mut buf);
    println!("{:?}", buf[0]);
    let lens: i32 = unsafe { mem::transmute(buf) };
    println!("{}", lens);
    

    /*let mut data: i32;
    let mut f = File::open("data/train_images").unwrap();
    f.read_be_i32().unwrap();
    unsafe { transmute(data.as_ptr()) };
    println!("{}", data);
	*/


}