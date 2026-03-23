
using Flux
using Statistics

# Simple neural network optimization example
function optimize_network(data, labels)
    model = Chain(
        Dense(10, 32, relu),
        Dense(32, 2, softmax)
    )
    loss(x, y) = Flux.crossentropy(model(x), y)
    ps = Flux.params(model)
    opt = ADAM()
    
    for epoch in 1:100
        Flux.train!(loss, ps, zip(data, labels), opt)
    end
    return model
end

println("NeuralFlux Optimization Library Initialized.")
