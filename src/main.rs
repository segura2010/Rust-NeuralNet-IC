
// Compilar y ejecutar con "cargo run" (no optimiza y es lento)

// Compilar con "cargo build --release" (optimiza)
// Ejecutar con "./target/release/redneuronal"


extern crate rand;
use rand::Rng;

use std::fs::File;
use std::io::Read;
use std::mem;
use std::ops::Rem;
use std::env;

#[warn(non_snake_case)]
struct Neuron
{
	pesos: Vec<f32>,
	bias: f32,
	error: f32,
	salida: f32,
	salidaSimple: f32 // sin aplicar la funcion de activacion
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

#[warn(non_snake_case)]
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
	    	let x = rand::random::<f32>() - 0.5;
	    	nn[0].push( Neuron{pesos:vec![x], error:0.0, salida:0.0, salidaSimple:0.0, bias:x} );
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
		    		let x = rand::random::<f32>() - 0.5;
		    		pesos.push(x);
		    	}
		    	let capaAct = nn.len()-1;
		    	nn[capaAct].push( Neuron{pesos:pesos, error:0.0, salida:0.0, salidaSimple:0.0, bias:0.5} );
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
	    		let x = rand::random::<f32>() - 0.5;
	    		pesos.push(x);
	    	}
	    	let capaAct = nn.len()-1;
	    	nn[capaAct].push( Neuron{pesos:pesos, error:0.0, salida:0.0, salidaSimple:0.0, bias:0.5} );
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
			o = o + self.capas[0][neuron].bias;
			self.capas[0][neuron].salidaSimple = o;
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
				o = o + self.capas[capa][neuron].bias;
				self.capas[capa][neuron].salidaSimple = o;
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
					println!("\tPeso {} // Salida {} // SalidaSimple {}", self.capas[c][n].pesos[p], self.capas[c][n].salida, self.capas[c][n].salidaSimple);
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
						// y actualizamos la "bias"
						self.capas[capaOculta][neuron].bias += self.tasaAprendizaje * self.capas[capaOculta][neuron].error;
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

fn invertir(en: &mut [u8]) -> [u8;4]
{
	let len = en.len();
	let mut buf = [0; 4];
	for i in 0..en.len()
	{
		let act = len - i - 1;
		buf[act] = en[i];
	}
	return buf;
}

fn encontrarMayor(en: Vec<f32>) -> usize
{
	let mut mayorIndice = 0 as usize;
	for i in 0..en.len()
	{
		if en[i] > en[mayorIndice]
		{
			mayorIndice = i;
		}
	}

	return mayorIndice;
}

fn main()
{

	// Ejemplo simple (XOR)
	// entradas y salidas
	let ent: Vec<Vec<f32> > = vec![vec![0f32, 0f32], vec![0f32, 1f32], vec![1f32, 0f32],vec![1f32, 1f32]];
	let sal: Vec<Vec<f32> > = vec![vec![1f32,0f32],vec![0f32,1f32], vec![0f32, 1f32], vec![1f32,0f32]];

	let entradas = ent[0].len() as i32;
	let salidas = sal[0].len() as i32;
	let neuronasOcultas = 50;
	let capasOcultas = 1;
	let epocas = 2000000;

	let mut nn = RedNeuronal::new(entradas, capasOcultas, neuronasOcultas, salidas, 0.6);

	nn.ejecutar(&ent[0]);
	println!("{:?}, {:?}", nn.salida(), encontrarMayor(nn.salida()));

	nn.entrenarBackPropagation(&ent, &sal, epocas);

	// salidas entrenadas
	println!("Salidas despues de entrenar:");
	nn.ejecutar(&ent[0]);
	println!("{:?}, {:?}", nn.salida(), encontrarMayor(nn.salida()));
	nn.ejecutar(&ent[1]);
	println!("{:?}, {:?}", nn.salida(), encontrarMayor(nn.salida()));
	nn.ejecutar(&ent[2]);
	println!("{:?}, {:?}", nn.salida(), encontrarMayor(nn.salida()));
	nn.ejecutar(&ent[3]);
	println!("{:?}, {:?}", nn.salida(), encontrarMayor(nn.salida()));
	

	/*
	// PROBLEMA MNIST
	// Lectura de ficheros
	let mut file=File::open("data/train_images").unwrap();
    let mut buf = [0; 4]; // buffer de 4 bytes
     
    // numero magico
    file.read(&mut buf); // leemos 4 bytes
    let mut p = invertir(&mut buf); // invertimos a bigendian
    let lens: i32 = unsafe { mem::transmute(p) }; // tranformamos a entero 32 bits
    println!("Numero Magico: {}", lens);

    // numero imagenes
    file.read(&mut buf); // leemos 4 bytes
    let numImagenes: i32 = unsafe { mem::transmute(invertir(&mut buf)) }; // tranformamos a entero 32 bits
    println!("Numero Imagenes: {}", numImagenes);

    // numero de filas y columnas
    file.read(&mut buf); // leemos 4 bytes
    let filas: i32 = unsafe { mem::transmute(invertir(&mut buf)) }; // tranformamos a entero 32 bits
    println!("Filas: {}", filas);

    file.read(&mut buf); // leemos 4 bytes
    let columnas: i32 = unsafe { mem::transmute(invertir(&mut buf)) }; // tranformamos a entero 32 bits
    println!("Columnas: {}", columnas);
    
    println!("Cargando imagenes ({})", numImagenes);
    // leemos las imagenes
    let mut pixelsLeidos = Vec::new();
    file.read_to_end(&mut pixelsLeidos);
    let mut imagenes: Vec< Vec<f32> > = vec![];
    let mut bufpixel = [0; 1]; // buffer de 4 bytes
    let pixels = (filas * columnas);

    
    for i in 0..pixelsLeidos.len()
    {
    	let modulo = (i as i32) % pixels;
    	if modulo == 0
    	{
    		imagenes.push(Vec::new());
    	}
    	let ultima = imagenes.len()-1;
    	imagenes[ultima].push((pixelsLeidos[i as usize] as f32)); // normalizando
    }
    println!("{:?}", imagenes.len());
	
	println!("Cargando etiquetas ({})", numImagenes);
	// Leemos las etiquetas
	let mut file=File::open("data/train_labels").unwrap();
	// saltamos numero magico y de imagenes
	file.read(&mut buf);
	file.read(&mut buf);
	// leemos las etiquetas de las imagenes
	let mut etiquetasLeidas = Vec::new();
    file.read_to_end(&mut etiquetasLeidas);
    let mut etiquetas: Vec< Vec<f32> > = vec![];
    let mut label = [0; 1]; // buffer de 4 bytes
    for imagen in 0..etiquetasLeidas.len()
    {	
		let ultima = imagenes.len() - 1;
		let mut v: Vec<f32> = vec![0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32];
		v[etiquetasLeidas[imagen] as usize] = 1f32;
		etiquetas.push(v);
	}
	println!("{:?}", etiquetas.len());

	let entradas = imagenes[0].len() as i32;
	let salidas = etiquetas[0].len() as i32;
	let neuronasOcultas = 3;
	let capasOcultas = 1;
	let epocas = 100;
	let tasa = 0.000000001;
	let mut red = RedNeuronal::new(entradas, capasOcultas, neuronasOcultas, salidas, tasa);

	println!("Entrenando.. (epocas: {}, tasa aprendizaje: {})", epocas, tasa);
	red.entrenarBackPropagation(&imagenes, &etiquetas, epocas);

	let probarCon = 44;
	red.ejecutar(&imagenes[probarCon]);
	let salidaBuena = encontrarMayor(etiquetas[probarCon].clone());
	let salidaRed = encontrarMayor(red.salida().clone());
	println!("Salida real: {:?}, {}", etiquetas[probarCon], salidaBuena);
	println!("Salida de la red: {:?}, {}", red.salida(), salidaRed);
	*/

}




