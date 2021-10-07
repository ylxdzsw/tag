a = """

"""

b = split(a, '\n')[1:end-1]

model_dict = Dict()

df = []
da = []
db = []
for f in b
    f0 = parse(Int, split(f)[1])
    if f0 < 817
        continue
    end

    gnumber = 0

    for line in eachline(split(f)[3])
        if startswith(line, '=')
            if gnumber != 0
                continue
            end

            gnumber = parse(Int, split(line, ' ')[5])
            if gnumber in keys(model_dict)
                push!(model_dict[gnumber], length(df)+1)
            else
                model_dict[gnumber] = [length(df)+1]
            end
            continue
        end

        try
            bf, aa, bb = split(line, ' ')
            # if abs(parse(Float64, aa) - parse(Float64, bb)) <= 1e-6
            #     continue
            # end
            push!(df, parse(Int, bf))
            push!(da, parse(Float64, aa))
            push!(db, parse(Float64, bb))
            break
        catch
        end
    end
end

println(model_dict)
println(df)
println(da)
println(db)
println()

for (key, value) in model_dict
    println(key, ' ', sum(value) / length(value))
end

cdf = sort!(da)
for (i, x) in enumerate(cdf)
    print("($(round(100x, digits=4)), $(round(i/length(cdf),digits=4))) ")
end
println()
println()

cdf = sort!(db)
for (i, x) in enumerate(cdf)
    print("($(round(100x, digits=4)), $(round(i/length(cdf),digits=4))) ")
end
println()
println()
